"""
Lab 06 — Evaluation block: memorization metrics.

What this covers:
    - compute_nn_stats: d1, d2, r = d1/d2 — what each means
    - compute_f_mem: the scalar phase boundary signal
    - compute_baseline_nn_stats: what "normal" (non-memorized) samples look like
    - extract_tau_mem: memorization onset timescale
    - fit_boundary: sigmoid fit to f_mem(log n) → n*
    - fit_tau_mem_scaling: τ_mem ∝ n^α (Bonnaire: α=1)
    - Synthetic experiments: memorization vs generalization regimes

Run from project root:
    python reference/walkthroughs/06_evaluation_block.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from experiments.synthetic.high_d.common.evaluation import (
    compute_baseline_nn_stats,
    compute_f_mem,
    compute_nn_stats,
    extract_tau_mem,
    fit_boundary,
    fit_tau_mem_scaling,
)
from experiments.synthetic.high_d.common.utils import seed_everything


def demo_nn_stats_intuition():
    print("\n── 1. NN stats intuition: memorization vs generalization ─────────")
    seed_everything(0)
    D, N_train = 16, 200

    training = torch.randn(N_train, D)

    # MEMORIZED: generated samples ≈ exact copies of training points
    gen_memorized = training[:50] + torch.randn(50, D) * 1e-5
    stats_m = compute_nn_stats(gen_memorized, training)
    f_mem_m = compute_f_mem(stats_m["r"], threshold=0.5)

    # GENERALIZED: generated samples are random (as if from true distribution)
    gen_random = torch.randn(50, D)
    stats_r = compute_nn_stats(gen_random, training)
    f_mem_r = compute_f_mem(stats_r["r"], threshold=0.5)

    print(f"  Training set: N={N_train}, D={D}")
    print()
    print(f"  Memorized samples (≈ copies of training):")
    print(f"    d1 (dist to nearest):  mean={stats_m['d1'].mean():.4f}")
    print(f"    d2 (dist to 2nd nearest): mean={stats_m['d2'].mean():.4f}")
    print(f"    r = d1/d2:             mean={stats_m['r'].mean():.4f}  (→ 0 = memorized)")
    print(f"    f_mem (r < 0.5):       {f_mem_m:.4f}  (→ 1 = all memorized)")
    print()
    print(f"  Generalized samples (random):")
    print(f"    d1:   mean={stats_r['d1'].mean():.4f}")
    print(f"    d2:   mean={stats_r['d2'].mean():.4f}")
    print(f"    r:    mean={stats_r['r'].mean():.4f}  (→ ~1 = generalized)")
    print(f"    f_mem:{f_mem_r:.4f}  (→ 0 = none memorized)")


def demo_threshold_sensitivity():
    print("\n── 2. f_mem threshold sensitivity ───────────────────────────────")
    seed_everything(0)
    D, N = 16, 100

    training = torch.randn(N, D)
    # Mix: 30% memorized, 70% generalized
    gen_mem = training[:30] + torch.randn(30, D) * 1e-5
    gen_gen = torch.randn(70, D)
    gen_mixed = torch.cat([gen_mem, gen_gen])

    stats = compute_nn_stats(gen_mixed, training)

    print(f"  Mixed generated set: 30% memorized copies + 70% random")
    print(f"  {'threshold':>12}  {'f_mem':>8}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        fm = compute_f_mem(stats["r"], threshold=thr)
        print(f"  {thr:>12.1f}  {fm:>8.4f}")
    print("  True rate = 0.30. Threshold shifts the estimate up/down.")
    print("  Relative ordering between parameterizations should be threshold-stable.")


def demo_baseline_comparison():
    print("\n── 3. Baseline comparison: generated vs held-out ─────────────────")
    seed_everything(0)
    D, N_train, M = 16, 100, 50

    training = torch.randn(N_train, D)
    # Held-out: drawn from SAME distribution as training
    holdout = torch.randn(M, D)

    baseline = compute_baseline_nn_stats(holdout, training)
    f_baseline = compute_f_mem(baseline["r"])

    # Memorizing model: generated ≈ copies
    gen_mem = training[:M] + torch.randn(M, D) * 1e-5
    stats_mem = compute_nn_stats(gen_mem, training)
    f_gen_mem = compute_f_mem(stats_mem["r"])

    # Generalizing model: generated from same Gaussian
    gen_good = torch.randn(M, D)
    stats_good = compute_nn_stats(gen_good, training)
    f_gen_good = compute_f_mem(stats_good["r"])

    print(f"  Training: N={N_train}, D={D}")
    print(f"  f_mem of held-out real samples:   {f_baseline:.4f}  ← baseline")
    print(f"  f_mem of memorizing model:         {f_gen_mem:.4f}  >> baseline → MEMORIZING")
    print(f"  f_mem of generalizing model:       {f_gen_good:.4f}  ≈ baseline → GENERALIZING")
    print()
    print("  Decision rule: if f_mem(generated) >> f_mem(baseline) → model memorizes.")


def demo_tau_mem():
    print("\n── 4. τ_mem: memorization onset timescale ────────────────────────")
    # Simulate typical checkpoint curves for different n
    def make_curve(n, n_star=500):
        """Synthetic f_mem curve: transitions from 1 to 0 around n_star."""
        epochs = list(range(10, 1010, 10))
        # τ_mem ∝ n (Bonnaire): model memorizes earlier for small n
        tau = int(50 * n / n_star)  # linear scaling
        f_mems = []
        for e in epochs:
            # Sigmoid transition in epoch: high early, low late
            f = 1 / (1 + np.exp(0.05 * (e - tau)))
            f_mems.append({"epoch": e, "f_mem": float(f)})
        return f_mems

    print(f"  Simulated checkpoint curves (τ_mem ∝ n, linear scaling):")
    n_values, tau_values = [], []
    for n in [100, 200, 500, 1000]:
        curve = make_curve(n)
        tau = extract_tau_mem(curve, onset_threshold=0.5)
        n_values.append(n)
        tau_values.append(tau)
        print(f"    n={n:>5}: τ_mem = {tau}  (memorization onset epoch)")

    # Fit scaling exponent α
    n_arr = np.array(n_values, dtype=float)
    tau_arr = np.array(tau_values, dtype=float)
    scaling = fit_tau_mem_scaling(n_arr, tau_arr)
    print(f"\n  τ_mem ∝ n^α fit:")
    print(f"    α = {scaling['alpha']:.3f}  (Bonnaire theory: α=1.0)")
    print(f"    R² = {scaling['r_squared']:.4f}")


def demo_boundary_fitting():
    print("\n── 5. Phase boundary: sigmoid fit to f_mem(log n) ───────────────")
    seed_everything(0)

    n_vals = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])

    # Simulate two parameterizations with different n* values
    def make_f_mem_curve(n_star, noise=0.03):
        f = 1 / (1 + np.exp(2.5 * (np.log10(n_vals) - np.log10(n_star))))
        f += np.random.randn(len(n_vals)) * noise
        return np.clip(f, 0, 1)

    np.random.seed(0)
    f_x   = make_f_mem_curve(n_star=200)   # x-pred: memorizes at smaller n
    f_eps = make_f_mem_curve(n_star=1000)  # eps-pred: needs more data to memorize

    print(f"  n:       " + "  ".join(f"{n:>6}" for n in n_vals))
    print(f"  f_mem(x-pred):   " + "  ".join(f"{v:>6.3f}" for v in f_x))
    print(f"  f_mem(eps-pred): " + "  ".join(f"{v:>6.3f}" for v in f_eps))

    res_x   = fit_boundary(n_vals, f_x)
    res_eps = fit_boundary(n_vals, f_eps)

    print(f"\n  Sigmoid fit results:")
    print(f"  x-pred:   n* = {res_x['n_star']:>8.1f}  R² = {res_x['r_squared']:.4f}  status={res_x['status']}")
    print(f"  eps-pred: n* = {res_eps['n_star']:>8.1f}  R² = {res_eps['r_squared']:.4f}  status={res_eps['status']}")
    print(f"\n  Ratio n*(eps) / n*(x) = {res_eps['n_star'] / res_x['n_star']:.2f}x")
    print("  This ratio is what the paper tests: does parameterization shift n*?")

    # Degenerate cases
    r1 = fit_boundary(n_vals, np.ones_like(n_vals, dtype=float) * 0.95)
    r2 = fit_boundary(n_vals, np.ones_like(n_vals, dtype=float) * 0.02)
    print(f"\n  Degenerate: all_memorize → status='{r1['status']}'")
    print(f"  Degenerate: all_generalize → status='{r2['status']}'")


if __name__ == "__main__":
    print("=" * 65)
    print("  Lab 06: Evaluation block (memorization metrics)")
    print("=" * 65)

    demo_nn_stats_intuition()
    demo_threshold_sensitivity()
    demo_baseline_comparison()
    demo_tau_mem()
    demo_boundary_fitting()

    print("\n" + "=" * 65)
    print("  Done. Evaluation block understood.")
    print("=" * 65)
