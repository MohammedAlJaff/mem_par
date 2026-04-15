"""
Lab 07 — Full mini pipeline: data → train → sample → evaluate.

This is the "lego set" assembled: one complete run of the experiment loop
using tiny settings so it finishes in < 1 minute on CPU.

What it demonstrates:
    1. Build a GMMSubspace dataset at a given (D, d, n)
    2. Build a TimeConditionedMLP + FlowMatching for a given (pred, loss)
    3. Train with Trainer (step-based)
    4. Generate samples using EMA weights (Heun integration)
    5. Compute memorization stats and f_mem
    6. Compare two parameterizations side-by-side

This is NOT a real experiment (too few steps/samples) — it's a smoke test
that all pieces fit together before you run the actual sweep.

Run from project root:
    python lab/07_full_pipeline.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from experiments.synthetic.high_d.common.data import GMMSubspace
from experiments.synthetic.high_d.common.evaluation import compute_f_mem, compute_nn_stats
from experiments.synthetic.high_d.common.flow_matching import FlowMatching
from experiments.synthetic.high_d.common.networks import TimeConditionedMLP
from experiments.synthetic.high_d.common.training import Trainer
from experiments.synthetic.high_d.common.utils import log_message, seed_everything


# ── Config for the mini run ────────────────────────────────────────────────
MINI_CFG = {
    "experiment": "lab07_mini",
    "training": {
        "lr": 3e-4,
        "batch_size": 64,
        "max_steps": 500,       # ~10s on CPU; use 5000+ for real results
        "ema_decay": 0.999,
        "warmup_fraction": 0.1,
        "weight_decay": 1e-4,
        "betas": [0.9, 0.95],
        "ckpt_interval_steps": 250,
    },
}

DATA_CFG = dict(D=64, d=4, K=10, sigma_z=0.5, sigma_eta=1e-3, n=150, geometry_seed=0)
NET_CFG  = dict(hidden_dim=128, depth=3, time_embed_dim=32)
N_SAMPLES = 200   # generated samples for evaluation
N_STEPS   = 30    # ODE steps (Heun) — enough for a mini demo


def build_and_train(pred_space: str, loss_space: str, data_seed: int, run_dir: str):
    """One complete train run. Returns (flow_model, ema, dataset)."""
    seed_everything(data_seed)

    # ── Data ─────────────────────────────────────────────────────────────
    dataset = GMMSubspace(**DATA_CFG, data_seed=data_seed)

    # ── Network + FlowMatching wrapper ───────────────────────────────────
    D = DATA_CFG["D"]
    net = TimeConditionedMLP(input_dim=D, **NET_CFG)
    flow_model = FlowMatching(net, pred_space=pred_space, loss_space=loss_space)

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(flow_model, dataset, MINI_CFG, run_dir=run_dir, device="cpu")
    summary = trainer.train()

    return flow_model, trainer.ema, dataset, summary


def generate_samples(flow_model: FlowMatching, ema, n: int, D: int) -> torch.Tensor:
    """Generate samples using EMA weights."""
    ema.apply()
    flow_model.eval()
    with torch.no_grad():
        samples = flow_model.sample((n, D), n_steps=N_STEPS, method="heun", device="cpu")
    ema.restore()
    return samples


def evaluate(generated: torch.Tensor, training_data: torch.Tensor, label: str) -> dict:
    """Compute and print memorization stats."""
    stats = compute_nn_stats(generated, training_data)
    f_mem = compute_f_mem(stats["r"], threshold=0.5)
    d1_mean = stats["d1"].mean().item()
    r_mean  = stats["r"].mean().item()

    print(f"  {label}:")
    print(f"    f_mem (r<0.5) = {f_mem:.4f}  {'← memorizing' if f_mem > 0.5 else '← generalizing'}")
    print(f"    d1 mean       = {d1_mean:.4f}  (dist to nearest training pt)")
    print(f"    r mean        = {r_mean:.4f}  (ratio d1/d2, 0=mem, 1=gen)")

    return {"f_mem": f_mem, "d1_mean": d1_mean, "r_mean": r_mean}


def main():
    print("=" * 65)
    print("  Lab 07: Full mini pipeline")
    print(f"  D={DATA_CFG['D']}, d={DATA_CFG['d']}, n={DATA_CFG['n']}, "
          f"steps={MINI_CFG['training']['max_steps']}")
    print("=" * 65)

    combos = [
        ("x",   "v",   "x-pred (Li&He)"),
        ("eps", "eps", "eps-pred (DDPM)"),
        ("v",   "v",   "v-pred (SD3)"),
    ]

    results = {}

    with tempfile.TemporaryDirectory() as base_dir:
        for pred, loss, label in combos:
            print(f"\n── Training: pred={pred}, loss={loss} ({'─'*20})")
            run_dir = str(Path(base_dir) / f"pred{pred}_loss{loss}")

            flow_model, ema, dataset, summary = build_and_train(
                pred, loss, data_seed=0, run_dir=run_dir
            )

            print(f"  final_loss = {summary['final_loss']:.6f}")
            print(f"  wall_time  = {summary['wall_time_seconds']:.1f}s")

            # ── Generate ──────────────────────────────────────────────────
            D = DATA_CFG["D"]
            samples = generate_samples(flow_model, ema, N_SAMPLES, D)
            print(f"  Generated {N_SAMPLES} samples via Heun({N_STEPS} steps)")
            print(f"  Sample stats: mean={samples.mean():.3f}, std={samples.std():.3f}")

            # ── Evaluate ──────────────────────────────────────────────────
            metrics = evaluate(samples, dataset.data, label)
            results[label] = metrics

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY: f_mem comparison across parameterizations")
    print("=" * 65)
    print(f"  (n={DATA_CFG['n']}, D={DATA_CFG['D']}, d={DATA_CFG['d']}, "
          f"steps={MINI_CFG['training']['max_steps']})\n")
    print(f"  {'label':>20}  {'f_mem':>8}  {'d1_mean':>10}  {'r_mean':>8}")
    for label, m in results.items():
        print(f"  {label:>20}  {m['f_mem']:>8.4f}  {m['d1_mean']:>10.4f}  {m['r_mean']:>8.4f}")

    print()
    print("  NOTE: With only 500 training steps these results are NOT meaningful.")
    print("  For real experiments use 10k-50k steps and multiple seeds.")
    print("  This script just verifies all pieces connect correctly.")
    print()
    print("  The canonical experiment flow (python -m experiments.gmm_subspace.train/evaluate/plot) does:")
    print("  1. Same structure but with local experiment config.yaml")
    print("  2. Full n sweep: n ∈ {50, 100, 200, 500, 1000, 2000, 5000, 10000}")
    print("  3. Multiple seeds for f_mem averaging")
    print("  4. Post-hoc eval via evaluation/evaluate_ckpts.py")
    print("  5. Boundary fitting via evaluation/memorization.py:fit_boundary()")


if __name__ == "__main__":
    main()
