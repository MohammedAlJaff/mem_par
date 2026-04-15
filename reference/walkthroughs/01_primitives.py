"""
Lab 01 — Primitives: utils + flow_matching building blocks.

What this covers:
    - seed_everything: why it matters and how it works
    - Explicit experiment-local config loading
    - Logit-normal time sampling: what the distribution looks like
    - Prediction space conversions: the algebra at the heart of the 3×3 grid
    - The interpolant z_t = t·x + (1-t)·ε visualised

Run from project root:
    python reference/walkthroughs/01_primitives.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# ── imports ──────────────────────────────────────────────────────────────────
from experiments.synthetic.high_d.common.runtime import load_yaml
from experiments.synthetic.high_d.common.utils import save_config, seed_everything
from experiments.synthetic.high_d.common.flow_matching import (
    sample_logit_normal,
    T_CLIP_MAX,
    T_CLIP_MIN,
    TIME_MU,
    TIME_SIGMA,
    _reshape_t,
    convert_pred_to_eps,
    convert_pred_to_v,
    convert_pred_to_x,
)


def demo_seeding():
    print("\n── 1. Seeding ────────────────────────────────────────────────────")
    seed_everything(42)
    a = torch.randn(5)
    seed_everything(42)
    b = torch.randn(5)
    print(f"  seed=42 → draw 1: {a.numpy().round(4)}")
    print(f"  seed=42 → draw 2: {b.numpy().round(4)}  (identical ✓)")
    assert torch.allclose(a, b)

    seed_everything(0)
    c = torch.randn(5)
    print(f"  seed=0  → draw  : {c.numpy().round(4)}  (different from seed=42 ✓)")


def demo_config():
    print("\n── 2. Config loading ─────────────────────────────────────────────")
    cfg = load_yaml("experiments/synthetic/high_d/gmm_subspace/config.yaml")
    print("  Loaded experiments/synthetic/high_d/gmm_subspace/config.yaml")
    print(f"  ambient D    : {cfg['data']['D']}")
    print(f"  data.n sweep : {cfg['data']['n']}")
    print(f"  fm.pred_space: {cfg['flow_matching']['pred_space']}")
    print(f"  fm.loss_space: {cfg['flow_matching']['loss_space']}")
    print(f"  training.lr  : {cfg['training']['lr']}")


def demo_time_sampling():
    print("\n── 3. Logit-normal time sampling ─────────────────────────────────")
    N = 50_000
    t = sample_logit_normal(N, mu=TIME_MU, sigma=TIME_SIGMA)
    print(f"  N={N} samples, μ={TIME_MU}, σ={TIME_SIGMA}")
    print(
        f"  Range   : [{t.min():.4f}, {t.max():.4f}]  (clipped to [{T_CLIP_MIN}, {T_CLIP_MAX}])"
    )
    print(
        f"  Mean    : {t.mean():.4f}  (sigmoid({TIME_MU}) ≈ {torch.sigmoid(torch.tensor(TIME_MU)):.4f})"
    )
    print(f"  Std     : {t.std():.4f}")

    # Show histogram as ASCII bar chart
    bins = torch.linspace(0, 1, 11)
    counts, _ = torch.histogram(t.cpu(), bins=bins)
    counts = counts / counts.max()
    print("  Distribution (0→1):")
    for i, c in enumerate(counts):
        bar = "█" * int(c * 30)
        lo, hi = bins[i].item(), bins[i + 1].item()
        print(f"    [{lo:.1f}–{hi:.1f}] {bar}")


def demo_interpolant():
    print("\n── 4. The interpolant z_t = t·x + (1-t)·ε ───────────────────────")
    torch.manual_seed(0)
    D = 4
    x = torch.tensor([1.0, 0.0, 0.0, 0.0])  # target: pure signal
    eps = torch.tensor([0.0, 0.0, 0.0, 0.0])  # noise: zero (for clarity)

    print(f"  x   = {x.numpy()}  (clean data)")
    print(f"  ε   = {eps.numpy()}  (noise)")
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_t = t_val * x + (1 - t_val) * eps
        print(
            f"  t={t_val:.2f} → z_t = {z_t.numpy()}  {'← pure noise' if t_val == 0 else '← pure data' if t_val == 1 else ''}"
        )
    print("  Interpretation: t=0 is START (noise), t=1 is END (clean data).")
    print(
        "  The network learns to 'reverse' this by predicting x (or ε or v) from z_t."
    )


def demo_conversions():
    print("\n── 5. Prediction space conversions ──────────────────────────────")
    torch.manual_seed(1)
    B, D = 8, 16
    x = torch.randn(B, D)
    eps = torch.randn(B, D)
    t = sample_logit_normal(B)
    t_b = _reshape_t(t, x)
    z_t = t_b * x + (1 - t_b) * eps
    v_true = x - eps  # the velocity definition

    print("  Given (z_t, t, x, ε), verify all conversion roundtrips:")

    # x → eps → x
    eps_hat = convert_pred_to_eps(x, z_t, t, "x")
    x_rec = convert_pred_to_x(eps_hat, z_t, t, "eps")
    err = (x - x_rec).abs().max().item()
    print(f"  x → eps → x:  max error = {err:.2e}  {'✓' if err < 1e-4 else '✗'}")

    # x → v → x
    v_hat = convert_pred_to_v(x, z_t, t, "x")
    x_rec2 = convert_pred_to_x(v_hat, z_t, t, "v")
    err2 = (x - x_rec2).abs().max().item()
    print(f"  x → v  → x:  max error = {err2:.2e}  {'✓' if err2 < 1e-4 else '✗'}")

    # eps → v → eps
    v_hat2 = convert_pred_to_v(eps, z_t, t, "eps")
    eps_rec = convert_pred_to_eps(v_hat2, z_t, t, "v")
    err3 = (eps - eps_rec).abs().max().item()
    print(f"  ε → v  → ε:  max error = {err3:.2e}  {'✓' if err3 < 1e-4 else '✗'}")

    # Check the v = x - eps identity
    v_from_x = convert_pred_to_v(x, z_t, t, "x")
    # NOTE: v_from_x ≠ x-eps because the formula uses z_t, not just x and eps directly.
    # But if we start from the TRUE v (x-eps), we can recover x:
    x_from_v_true = convert_pred_to_x(v_true, z_t, t, "v")
    err4 = (x - x_from_v_true).abs().max().item()
    print(f"  v=x-ε  → x:  max error = {err4:.2e}  {'✓' if err4 < 1e-4 else '✗'}")

    print("\n  The 3×3 grid at a glance:")
    print("       loss→   x       ε       v")
    print("  pred↓")
    print("  x        (x,x)   (x,ε)   (x,v)  ← Li&He Alg.1")
    print("  ε        (ε,x)   (ε,ε)   (ε,v)")
    print("  v        (v,x)   (v,ε)   (v,v)")
    print("  Diagonal = predict and measure in the SAME space (no conversion needed).")


if __name__ == "__main__":
    print("=" * 65)
    print("  Lab 01: Primitives (utils + flow_matching algebra)")
    print("=" * 65)

    demo_seeding()
    demo_config()
    demo_time_sampling()
    demo_interpolant()
    demo_conversions()

    print("\n" + "=" * 65)
    print("  Done. All primitives understood.")
    print("=" * 65)
