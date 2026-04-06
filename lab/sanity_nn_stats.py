"""
Sanity-check compute_nn_stats and compute_f_mem with controlled inputs.

We build a small reference set, then create "generated" points via
interpolation with a tunable parameter alpha:

    g_i = alpha * ref[i] + (1 - alpha) * ref[j]

alpha = 1.0  =>  exact copies          =>  r ~ 0,  f_mem ~ 1
alpha = 0.5  =>  midpoint between two   =>  r ~ 1,  f_mem ~ 0
alpha = 0.99 =>  near-copies            =>  r small, f_mem high

Run:
    python lab/sanity_nn_stats.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from experiments.natural_images.common.evaluation.memorization import (
    compute_nn_stats,
    compute_f_mem,
)


def make_reference(n: int, d: int, seed: int = 0) -> torch.Tensor:
    """Well-separated reference points (unit-variance Gaussian)."""
    torch.manual_seed(seed)
    return torch.randn(n, d)


def make_interpolated(
    ref: torch.Tensor,
    alpha: float,
    m: int | None = None,
    seed: int = 1,
) -> torch.Tensor:
    """
    For each of m generated points, pick two distinct reference indices
    and return:  g = alpha * ref[i] + (1 - alpha) * ref[j]

    alpha = 1.0  ->  exact copy of ref[i]
    alpha = 0.5  ->  midpoint
    alpha = 0.0  ->  exact copy of ref[j]
    """
    n = len(ref)
    if m is None:
        m = n
    rng = np.random.default_rng(seed)
    idx_i = rng.integers(0, n, size=m)
    idx_j = rng.integers(0, n, size=m)
    # make sure j != i so the interpolation is non-trivial
    mask = idx_i == idx_j
    idx_j[mask] = (idx_j[mask] + 1) % n
    return alpha * ref[idx_i] + (1.0 - alpha) * ref[idx_j]


def run_one(ref: torch.Tensor, gen: torch.Tensor, label: str, threshold: float = 0.5):
    stats = compute_nn_stats(gen, ref)
    f = compute_f_mem(stats["r"], threshold=threshold)
    print(f"  {label:30s}  |  f_mem={f:.3f}  mean_r={stats['r'].mean():.4f}  "
          f"median_r={stats['r'].median():.4f}  min_r={stats['r'].min():.4f}  "
          f"max_r={stats['r'].max():.4f}")
    return stats, f


def main():
    N, D = 50, 16
    M = 200
    ref = make_reference(N, D)

    print(f"Reference: {N} points in R^{D}")
    print(f"Generated: {M} points per test")
    print(f"{'':32s}  |  {'f_mem':>5s}  {'mean_r':>8s}  {'med_r':>8s}  {'min_r':>8s}  {'max_r':>8s}")
    print("-" * 100)

    # ---- Sweep alpha from exact copy to midpoint ----
    alphas = [1.0, 0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    results = []
    for a in alphas:
        gen = make_interpolated(ref, alpha=a, m=M)
        _, f = run_one(ref, gen, f"alpha={a:.3f}")
        results.append((a, f))

    # ---- Pure noise (no relationship to ref) ----
    print("-" * 100)
    torch.manual_seed(99)
    noise = torch.randn(M, D)
    run_one(ref, noise, "pure noise (no relation)")

    # ---- Exact copies + small Gaussian jitter ----
    print("-" * 100)
    for sigma in [0.0, 1e-4, 1e-2, 0.1, 0.5]:
        torch.manual_seed(7)
        idx = torch.randint(0, N, (M,))
        gen = ref[idx] + sigma * torch.randn(M, D)
        run_one(ref, gen, f"copy + noise(sigma={sigma})")

    # ---- Summary ----
    print("\n=== Quick reference ===")
    print("alpha=1.0 (exact copy)   -> f_mem should be ~1, r ~ 0")
    print("alpha=0.5 (midpoint)     -> f_mem should be ~0, r ~ 1")
    print("pure noise               -> f_mem should be ~0, r ~ 1")
    print("copy+tiny noise          -> f_mem should be ~1, r ~ 0")


if __name__ == "__main__":
    main()
