"""
Lab 05 — Training blocks: EMA, LR schedule, and Trainer.

What this covers:
    - EMA: how shadow weights drift away from training weights, why it matters
    - CosineWarmup: the LR profile, warmup vs cosine decay
    - Trainer: running a tiny end-to-end training loop (step-based)
    - Checkpoint structure: what gets saved and why

Run from project root:
    python lab/05_training_blocks.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from experiments.synthetic.high_d.common.data import GMMSubspace
from experiments.synthetic.high_d.common.flow_matching import FlowMatching
from experiments.synthetic.high_d.common.networks import TimeConditionedMLP
from experiments.synthetic.high_d.common.training import CosineWarmup, EMA, Trainer
from experiments.synthetic.high_d.common.utils import seed_everything


def demo_ema():
    print("\n── 1. EMA (Exponential Moving Average) ───────────────────────────")
    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    ema = EMA(model, decay=0.9)

    # Track divergence between model and shadow over fake training
    diffs = []
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(100):
        # Fake gradient step: push weights toward 1.0
        loss = ((model.weight - 1.0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update()

        if step % 20 == 0:
            diff = (model.weight.data - ema.shadow["weight"]).abs().mean().item()
            diffs.append((step, diff, model.weight.data.mean().item()))

    print(f"  decay=0.9  (effective window ≈ 1/(1-0.9) = 10 steps)")
    print(f"  {'step':>6}  {'|w - shadow|':>14}  {'w.mean':>8}")
    for step, diff, wmean in diffs:
        print(f"  {step:>6}  {diff:>14.6f}  {wmean:>8.4f}")
    print("  Shadow lags behind the training weights → smoother, lower-variance params.")

    # Demonstrate apply/restore
    w_before = model.weight.data.clone()
    ema.apply()
    w_ema = model.weight.data.clone()
    ema.restore()
    w_after = model.weight.data.clone()
    print(f"\n  Before apply: weight[0,0] = {w_before[0,0]:.6f}")
    print(f"  After  apply: weight[0,0] = {w_ema[0,0]:.6f}  (EMA value)")
    print(f"  After restore: weight[0,0]= {w_after[0,0]:.6f}  (back to training weights)")


def demo_lr_schedule():
    print("\n── 2. CosineWarmup LR schedule ───────────────────────────────────")
    LR = 3e-4
    total, warmup = 10_000, 500
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=LR)
    sched = CosineWarmup(opt, total_steps=total, warmup_steps=warmup)

    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()

    lrs = torch.tensor(lrs)
    # Print profile at key checkpoints
    checkpoints = [0, warmup//2, warmup, total//4, total//2, 3*total//4, total-1]
    print(f"  LR_max={LR:.0e}, total={total}, warmup={warmup} ({warmup/total:.0%})")
    print(f"\n  {'step':>7}  {'LR':>12}  {'phase'}")
    for s in checkpoints:
        phase = "warmup" if s < warmup else "cosine decay"
        print(f"  {s:>7}  {lrs[s].item():>12.2e}  {phase}")

    peak_step = lrs.argmax().item()
    print(f"\n  Peak LR at step {peak_step}: {lrs[peak_step].item():.2e}  (= LR_max ✓)")
    print(f"  Final LR: {lrs[-1].item():.2e}  (decayed to ~0 ✓)")


def demo_trainer_minimal():
    print("\n── 3. Trainer: minimal end-to-end run ────────────────────────────")
    seed_everything(0)

    # Tiny dataset and model for fast testing
    D, d, n = 32, 4, 100
    dataset = GMMSubspace(D=D, d=d, K=5, n=n, geometry_seed=0, data_seed=0)

    net = TimeConditionedMLP(input_dim=D, hidden_dim=64, depth=2, time_embed_dim=32)
    flow_model = FlowMatching(net, pred_space="x", loss_space="v")

    config = {
        "experiment": "lab_test",
        "training": {
            "lr": 3e-4,
            "batch_size": 32,
            "max_steps": 200,          # small: just to verify it runs
            "ema_decay": 0.999,
            "warmup_fraction": 0.1,
            "weight_decay": 1e-4,
            "betas": [0.9, 0.95],
            "ckpt_interval_steps": 100,
        },
    }

    with tempfile.TemporaryDirectory() as run_dir:
        trainer = Trainer(flow_model, dataset, config, run_dir=run_dir, device="cpu")
        print(f"  Dataset: n={n}, D={D}, d={d}")
        print(f"  Model:   pred=x, loss=v, W=64, depth=2")
        print(f"  Training for {config['training']['max_steps']} steps on CPU...")

        summary = trainer.train()

        print(f"\n  Training summary:")
        print(f"    final_loss   = {summary['final_loss']:.6f}")
        print(f"    total_steps  = {summary['total_steps']}")
        print(f"    wall_time    = {summary['wall_time_seconds']:.1f}s")

        # Check what was saved
        run_path = Path(run_dir)
        ckpt_dir = run_path / "checkpoints"
        saved = sorted(ckpt_dir.glob("*.pt"))
        print(f"\n  Checkpoints saved:")
        for ck in saved:
            print(f"    {ck.name}")

        # Inspect final checkpoint structure
        if saved:
            ck = torch.load(saved[-1], map_location="cpu", weights_only=False)
            print(f"\n  Checkpoint keys: {list(ck.keys())}")
            print(f"  global_step = {ck['global_step']}")

    print("\n  ✓ Trainer runs end-to-end and produces checkpoints.")


def demo_ema_in_training():
    print("\n── 4. EMA in training: train weights vs EMA weights ─────────────")
    seed_everything(0)
    D, n = 16, 80

    dataset = GMMSubspace(D=D, d=4, K=5, n=n, geometry_seed=0, data_seed=0)
    net = TimeConditionedMLP(input_dim=D, hidden_dim=32, depth=2, time_embed_dim=16)
    flow_model = FlowMatching(net, pred_space="v", loss_space="v")

    config = {
        "experiment": "lab_ema_test",
        "training": {
            "lr": 1e-3,
            "batch_size": 32,
            "max_steps": 300,
            "ema_decay": 0.99,       # fast EMA for demonstration
            "warmup_fraction": 0.05,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
        },
    }

    x_all = dataset.data

    with tempfile.TemporaryDirectory() as run_dir:
        trainer = Trainer(flow_model, dataset, config, run_dir=run_dir, device="cpu")
        trainer.train()

        # Compare train weights vs EMA weights on a held-out batch
        flow_model.eval()
        test_batch = torch.randn(32, D)

        with torch.no_grad():
            loss_train = flow_model.compute_loss(test_batch).item()
            trainer.ema.apply()
            loss_ema = flow_model.compute_loss(test_batch).item()
            trainer.ema.restore()

    print(f"  After {config['training']['max_steps']} steps (decay=0.99):")
    print(f"  Train-weight loss  = {loss_train:.6f}")
    print(f"  EMA-weight loss    = {loss_ema:.6f}")
    print("  EMA weights typically have slightly lower loss (smoother landscape).")
    print("  For final evaluation, ALWAYS use EMA weights.")


if __name__ == "__main__":
    print("=" * 65)
    print("  Lab 05: Training blocks (EMA + LR schedule + Trainer)")
    print("=" * 65)

    demo_ema()
    demo_lr_schedule()
    demo_trainer_minimal()
    demo_ema_in_training()

    print("\n" + "=" * 65)
    print("  Done. Training blocks understood.")
    print("=" * 65)
