"""
Memorization metrics and phase boundary estimation.

This is the measurement core of the paper. Everything we claim about
parameterization shifting the memorization boundary depends on these
metrics being correct and well-calibrated.

Metrics (weakest to strongest):
    1. d1 (1-NN distance): How close is each generated sample to the
       nearest training point? Small d1 → sample is near training data.
       PROBLEM: small d1 also happens for good generalization in dense
       regions. Ambiguous on its own.

    2. r = d1/d2 (NN ratio): How close is the NEAREST training point
       relative to the SECOND nearest? Small r → the sample is
       suspiciously close to ONE specific training point. This is the
       signature of memorization: copying a single example rather than
       interpolating between examples.

    3. f_mem (memorization fraction): Fraction of generated samples
       with r < threshold. This is the scalar summary statistic we
       plot against n to find the phase boundary.

    4. Baseline comparison: Compare f_mem of generated samples against
       f_mem of real held-out samples measured against the same training
       set. If they're similar, the model is generalizing (its NN stats
       look like those of the true distribution). If f_mem is much higher
       for generated samples, it's memorizing.

Phase boundary estimation:
    We fit a sigmoid to f_mem(log n) and extract the inflection point
    as n* (the critical dataset size). The sigmoid model is:
        f(x) = L / (1 + exp(-k(x - x0))) + b
    where x = log10(n), x0 is the midpoint, and n* = 10^x0.

    This is a smooth approximation to what is often a fairly sharp
    transition. The fit quality (R²) tells us whether the transition
    is actually sigmoid-shaped or something else.

τ_mem extraction:
    For each fixed n, we track f_mem across training checkpoints.
    τ_mem is the first epoch/step where f_mem exceeds an onset threshold.
    Bonnaire et al. (2025) showed τ_mem ∝ n (linear scaling).
    We test whether this scaling changes with parameterization.
"""

import torch
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from typing import Optional


# =============================================================================
# Core NN statistics
# =============================================================================


def compute_nn_stats(
    generated: torch.Tensor,
    training: torch.Tensor,
) -> dict:
    """
    Compute nearest-neighbor statistics in Euclidean space.

    For each generated sample, find its 1st and 2nd nearest neighbors
    in the training set. Return distances and the ratio d1/d2.

    USE THIS FOR: synthetic data, CelebA 32×32 greyscale (pixel space)
    USE LPIPSNNComputer FOR: CelebA 64+ RGB, AFHQ, Flowers (perceptual space)

    Args:
        generated: (M, D) generated samples (flattened if images)
        training:  (N, D) training samples (flattened if images)

    Returns dict:
        d1: (M,) distance to nearest training neighbor
        d2: (M,) distance to second nearest training neighbor
        r:  (M,) ratio d1/d2 — the PRIMARY memorization signal
        nn_idx: (M,) index of nearest training point (for visualization)

    Interpretation of r:
        r ≈ 0:   sample is sitting on top of a single training point → memorized
        r ≈ 0.5: sample is closer to one point than its neighbor → possibly memorized
        r ≈ 1.0: sample is roughly equidistant to multiple points → likely generalized
        r > 1.0: impossible by definition (d1 ≤ d2)
    """
    # Flatten to 2D if needed (e.g., images: (M, C, H, W) → (M, C*H*W))
    gen_flat = generated.reshape(generated.shape[0], -1).numpy()
    train_flat = training.reshape(training.shape[0], -1).numpy()

    # Build KD-tree for fast NN lookup
    # KDTree is O(N log N) to build, O(log N) per query.
    # For N = 50k and M = 10k, this is fast enough (<1 sec).
    tree = KDTree(train_flat)
    dists, idxs = tree.query(gen_flat, k=2)

    d1 = torch.tensor(dists[:, 0], dtype=torch.float32)
    d2 = torch.tensor(dists[:, 1], dtype=torch.float32)

    # Avoid division by zero (happens when two training points coincide)
    r = d1 / (d2 + 1e-10)

    return {
        "d1": d1,
        "d2": d2,
        "r": r,
        "nn_idx": torch.tensor(idxs[:, 0], dtype=torch.long),
    }


def compute_f_mem(
    r: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Memorization fraction: proportion of samples with NN ratio below threshold.

    This is the scalar we plot against n to find the phase boundary.

    Args:
        r: (M,) NN ratios from compute_nn_stats
        threshold: cutoff for "memorized" classification
            0.3 → strict (only very clear copies count)
            0.5 → standard (our default, following Yoon et al.)
            0.7 → lenient (catches more ambiguous cases)

            The choice of threshold shifts f_mem up or down but should NOT
            change the relative ordering between parameterizations. If it
            does, that's a red flag — report as sensitivity analysis.

    Returns:
        float in [0, 1] — fraction of memorized samples
    """
    return (r < threshold).float().mean().item()


# =============================================================================
# Baseline comparison
# =============================================================================


def compute_baseline_nn_stats(
    holdout: torch.Tensor,
    training: torch.Tensor,
) -> dict:
    """
    Compute NN stats for REAL held-out data against the training set.

    This gives the baseline: what do non-memorized samples look like
    when compared to the training set? If generated samples have similar
    stats, the model is generalizing.

    The key comparison:
        f_mem(generated vs training)  vs  f_mem(holdout vs training)
        If similar → model generalizes
        If generated >> holdout → model memorizes

    Args:
        holdout:  (M_ref, D) held-out real samples
        training: (N, D) training samples

    Returns:
        Same format as compute_nn_stats
    """
    return compute_nn_stats(holdout, training)


# =============================================================================
# Phase boundary fitting
# =============================================================================


def fit_boundary(
    n_values: np.ndarray,
    f_mem_values: np.ndarray,
    f_mem_std: Optional[np.ndarray] = None,
) -> dict:
    """
    Fit a sigmoid to f_mem vs log(n), extract n* as the inflection point.

    The sigmoid model:
        f(x) = L / (1 + exp(-k * (x - x0))) + b

    where x = log10(n). The inflection point x0 corresponds to n* = 10^x0.

    The sigmoid is a natural model for a phase transition: f_mem ≈ 1 for
    small n (full memorization), decreases through a transition region,
    and reaches f_mem ≈ baseline for large n (full generalization).
    The steepness k tells us how sharp the transition is.

    CAVEATS:
        - If the transition is not sigmoid-shaped, the fit will be poor
          (check R²). In that case, use an alternative like interpolation.
        - If all f_mem values are high (all memorization), we can't fit
          a transition — report n* > max(n).
        - If all f_mem values are low (all generalization), report n* < min(n).

    Args:
        n_values:    (K,) dataset sizes
        f_mem_values: (K,) memorization fractions (mean over seeds)
        f_mem_std:   (K,) optional std for weighting

    Returns dict:
        n_star:    estimated critical dataset size
        n_star_ci: (low, high) 95% bootstrap confidence interval
        params:    {L, k, x0, b} sigmoid parameters
        r_squared: goodness of fit
        status:    'ok', 'all_memorize', 'all_generalize', 'fit_failed'
    """
    x = np.log10(n_values.astype(float))
    y = f_mem_values.astype(float)

    # Check for degenerate cases
    if np.all(y > 0.8):
        return {
            "n_star": float("inf"),
            "n_star_ci": (float("nan"), float("nan")),
            "params": {},
            "r_squared": float("nan"),
            "status": "all_memorize",
        }
    if np.all(y < 0.1):
        return {
            "n_star": 0.0,
            "n_star_ci": (float("nan"), float("nan")),
            "params": {},
            "r_squared": float("nan"),
            "status": "all_generalize",
        }

    # Sigmoid model
    def sigmoid(x, L, k, x0, b):
        return L / (1.0 + np.exp(-k * (x - x0))) + b

    try:
        # Initial guess: L≈1 (full range), k<0 (decreasing), x0=median
        p0 = [1.0, -3.0, np.median(x), 0.0]
        bounds = (
            [0.0, -20.0, x.min() - 1, -0.5],  # lower bounds
            [1.5, 0.0, x.max() + 1, 0.5],  # upper bounds
        )

        # Optional weighting by inverse variance
        sigma = 1.0 / (f_mem_std + 1e-6) if f_mem_std is not None else None

        popt, pcov = curve_fit(
            sigmoid, x, y, p0=p0, bounds=bounds, sigma=sigma, maxfev=5000
        )
        L, k, x0, b = popt
        n_star = 10.0**x0

        # R² (goodness of fit)
        y_pred = sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-10)

        # Bootstrap confidence interval for n*
        n_star_bootstrap = []
        for _ in range(1000):
            idx = np.random.choice(len(x), len(x), replace=True)
            try:
                pb, _ = curve_fit(
                    sigmoid, x[idx], y[idx], p0=popt, bounds=bounds, maxfev=2000
                )
                n_star_bootstrap.append(10.0 ** pb[2])
            except (RuntimeError, ValueError):
                continue

        if len(n_star_bootstrap) > 50:
            ci = tuple(np.percentile(n_star_bootstrap, [2.5, 97.5]).tolist())
        else:
            ci = (float("nan"), float("nan"))

        return {
            "n_star": float(n_star),
            "n_star_ci": ci,
            "params": {"L": float(L), "k": float(k), "x0": float(x0), "b": float(b)},
            "r_squared": float(r_squared),
            "status": "ok",
        }

    except (RuntimeError, ValueError) as e:
        return {
            "n_star": float("nan"),
            "n_star_ci": (float("nan"), float("nan")),
            "params": {},
            "r_squared": float("nan"),
            "status": f"fit_failed: {str(e)}",
        }


# =============================================================================
# τ_mem extraction
# =============================================================================


def extract_tau_mem(
    checkpoint_metrics: list,
    onset_threshold: float = 0.05,
) -> Optional[int]:
    """
    Extract memorization timescale from checkpoint-level f_mem values.

    τ_mem = first epoch/step where f_mem exceeds onset_threshold.
    Bonnaire et al. showed τ_mem ∝ n. We test whether this scaling
    changes with parameterization.

    Args:
        checkpoint_metrics: list of dicts, each with keys:
            'epoch' (or 'step'): training progress
            'f_mem': memorization fraction at this checkpoint
            Must be sorted by epoch/step.

        onset_threshold: f_mem value defining memorization onset
            0.01 → very early detection (conservative)
            0.05 → standard (Bonnaire's choice)
            0.10 → late detection (lenient)

    Returns:
        The epoch/step value at first onset, or None if never reached.

    NOTE: τ_mem depends on checkpoint granularity. If checkpoints are
    every 50 epochs, τ_mem has ±50 epoch uncertainty. Finer checkpointing
    gives better resolution but costs more disk space and eval time.
    """
    # Sort by training progress (epoch or step)
    key = "epoch" if "epoch" in checkpoint_metrics[0] else "step"
    sorted_metrics = sorted(checkpoint_metrics, key=lambda m: m[key])

    for entry in sorted_metrics:
        if entry["f_mem"] > onset_threshold:
            return entry[key]

    return None  # memorization never detected


def fit_tau_mem_scaling(
    n_values: np.ndarray,
    tau_mem_values: np.ndarray,
) -> dict:
    """
    Fit τ_mem ∝ n^α and extract the scaling exponent α.

    Bonnaire et al. predict α = 1 (linear scaling). Deviations from
    linearity would indicate that the memorization dynamics are more
    complex than the RF theory predicts.

    We fit log(τ_mem) = α · log(n) + c via linear regression.

    Args:
        n_values: (K,) dataset sizes where τ_mem was measured
        tau_mem_values: (K,) corresponding τ_mem values

    Returns dict:
        alpha: scaling exponent (Bonnaire predicts 1.0)
        alpha_ci: 95% CI from bootstrap
        intercept: log-space intercept
        r_squared: fit quality
    """
    # Filter out NaN / inf
    mask = np.isfinite(tau_mem_values) & (tau_mem_values > 0)
    if mask.sum() < 3:
        return {
            "alpha": float("nan"),
            "alpha_ci": (float("nan"), float("nan")),
            "intercept": float("nan"),
            "r_squared": float("nan"),
        }

    log_n = np.log10(n_values[mask])
    log_tau = np.log10(tau_mem_values[mask])

    # Linear regression in log-log space
    from numpy.polynomial import polynomial as P

    coeffs = np.polyfit(log_n, log_tau, 1)  # [slope, intercept]
    alpha = coeffs[0]
    intercept = coeffs[1]

    # R²
    log_tau_pred = alpha * log_n + intercept
    ss_res = np.sum((log_tau - log_tau_pred) ** 2)
    ss_tot = np.sum((log_tau - log_tau.mean()) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-10)

    # Bootstrap CI for alpha
    alphas = []
    for _ in range(1000):
        idx = np.random.choice(len(log_n), len(log_n), replace=True)
        c = np.polyfit(log_n[idx], log_tau[idx], 1)
        alphas.append(c[0])
    ci = tuple(np.percentile(alphas, [2.5, 97.5]).tolist())

    return {
        "alpha": float(alpha),
        "alpha_ci": ci,
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


# =============================================================================
# Self-test  (run with: python -m src.evaluation.memorization  from project root)
# =============================================================================

if __name__ == "__main__":
    print("=== experiments.synthetic.low_d.common.evaluation.memorization self-test ===\n")
    torch.manual_seed(0)
    np.random.seed(0)

    D, N_train, M_gen = 8, 50, 30

    # --- compute_nn_stats: memorized samples ---
    # Create generated samples that are copies of training samples → should have r ≈ 0
    training = torch.randn(N_train, D)
    gen_memorized = training[:M_gen] + torch.randn(M_gen, D) * 1e-6  # tiny jitter
    stats_mem = compute_nn_stats(gen_memorized, training)
    assert stats_mem["d1"].shape == (M_gen,)
    assert stats_mem["d2"].shape == (M_gen,)
    assert stats_mem["r"].shape == (M_gen,)
    f_mem_high = compute_f_mem(stats_mem["r"], threshold=0.5)
    assert f_mem_high > 0.9, f"Near-copy samples should have f_mem > 0.9, got {f_mem_high:.3f}"
    print(f"[PASS] Near-copy samples → f_mem = {f_mem_high:.3f}  (expect > 0.9)")

    # --- compute_nn_stats: generalizing samples (interpolations) ---
    # Interpolated samples between training points → r closer to 1
    idx1 = torch.randint(0, N_train, (M_gen,))
    idx2 = torch.randint(0, N_train, (M_gen,))
    alpha = 0.5
    gen_interp = alpha * training[idx1] + (1 - alpha) * training[idx2]
    stats_gen = compute_nn_stats(gen_interp, training)
    f_mem_low = compute_f_mem(stats_gen["r"], threshold=0.5)
    # Interpolations need not be < 0.5; r is d1/(d2+eps) so at midpoint r ≈ 1.
    # Just check r > 0 and < 1 (valid range)
    assert (stats_gen["r"] >= 0).all() and (stats_gen["r"] <= 1 + 1e-5).all()
    print(f"[PASS] Interpolated samples → r ∈ [0, 1], f_mem = {f_mem_low:.3f}")

    # --- compute_baseline_nn_stats: same as compute_nn_stats ---
    holdout = torch.randn(M_gen, D)
    baseline = compute_baseline_nn_stats(holdout, training)
    assert "r" in baseline
    print("[PASS] compute_baseline_nn_stats returns same structure as compute_nn_stats")

    # --- extract_tau_mem ---
    ckpt_metrics = [
        {"epoch": 10, "f_mem": 0.01},
        {"epoch": 20, "f_mem": 0.03},
        {"epoch": 30, "f_mem": 0.07},  # ← first above 0.05
        {"epoch": 40, "f_mem": 0.20},
    ]
    tau = extract_tau_mem(ckpt_metrics, onset_threshold=0.05)
    assert tau == 30, f"Expected τ_mem=30, got {tau}"
    print(f"[PASS] extract_tau_mem: τ_mem = {tau}  (first epoch with f_mem > 0.05)")

    tau_never = extract_tau_mem([{"epoch": 10, "f_mem": 0.01}], onset_threshold=0.5)
    assert tau_never is None
    print("[PASS] extract_tau_mem returns None when threshold never reached")

    # --- fit_boundary ---
    # Synthetic f_mem(n): high for small n, low for large n
    n_vals = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    # True n* ≈ 500 (midpoint in log space)
    f_mem_vals = 1.0 / (1.0 + np.exp(3.0 * (np.log10(n_vals) - np.log10(500))))
    result = fit_boundary(n_vals, f_mem_vals)
    assert result["status"] == "ok", f"fit_boundary failed: {result['status']}"
    assert abs(result["n_star"] - 500) / 500 < 0.2, f"n* = {result['n_star']:.1f}, expected ~500"
    print(f"[PASS] fit_boundary: n* = {result['n_star']:.1f}  (true ~500), R² = {result['r_squared']:.4f}")

    # Degenerate cases
    r_all_mem = fit_boundary(n_vals, np.ones_like(f_mem_vals) * 0.95)
    assert r_all_mem["status"] == "all_memorize"
    r_all_gen = fit_boundary(n_vals, np.ones_like(f_mem_vals) * 0.02)
    assert r_all_gen["status"] == "all_generalize"
    print("[PASS] fit_boundary detects degenerate all_memorize / all_generalize cases")

    # --- fit_tau_mem_scaling ---
    # If τ_mem ∝ n^1, we expect α ≈ 1
    n_arr = np.array([100.0, 200, 500, 1000, 2000])
    tau_arr = 0.5 * n_arr  # true α = 1
    scaling = fit_tau_mem_scaling(n_arr, tau_arr)
    assert abs(scaling["alpha"] - 1.0) < 0.05, f"Expected α≈1, got {scaling['alpha']:.3f}"
    print(f"[PASS] fit_tau_mem_scaling: α = {scaling['alpha']:.4f}  (true α=1), R² = {scaling['r_squared']:.4f}")

    print("\n=== All memorization tests passed ===")
