"""
Cosine learning rate schedule with linear warmup.

WHY THIS SCHEDULE?
    Linear warmup: prevents early instability when the loss landscape
    is poorly conditioned (which it is at initialization). Standard
    practice for transformers and modern diffusion models.

    Cosine decay: smoothly anneals the LR to 0 over training. Less
    aggressive than step decay (no sudden drops), and avoids the
    "learning rate too high at the end" problem of constant LR.
    Li & He actually use constant LR, but cosine is safer for the
    smaller datasets we use (less overfitting risk).

    The full schedule:
        step < warmup_steps:  lr = lr_max × (step / warmup_steps)
        step ≥ warmup_steps:  lr = lr_max × 0.5 × (1 + cos(π × progress))
    where progress = (step - warmup_steps) / (total_steps - warmup_steps)

    Paper spec: warmup_fraction = 0.05 (first 5% of training)

    Alternative: constant LR (Li & He use lr=2e-4 constant)
        Pro: simpler, one less hyperparameter
        Con: risk of overfitting on small datasets (our setting)
    Alternative: step decay (halve LR at 50% and 75% of training)
        Pro: simple, well-understood
        Con: discontinuities can destabilize training dynamics
    Alternative: linear decay
        Pro: simple
        Con: less theoretically motivated than cosine for SGD convergence
"""

import math
from torch.optim.lr_scheduler import LambdaLR


class CosineWarmup(LambdaLR):
    """
    Cosine annealing with linear warmup.

    Usage:
        scheduler = CosineWarmup(optimizer, total_steps=10000, warmup_steps=500)
        for step in range(total_steps):
            optimizer.step()
            scheduler.step()

    Args:
        optimizer: torch optimizer
        total_steps: total training steps
        warmup_steps: linear warmup period
        min_lr_fraction: minimum LR as fraction of peak (0 = decay to 0)
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr_fraction: float = 0.0,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_fraction = min_lr_fraction

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup: 0 → 1
                return step / max(1, warmup_steps)
            else:
                # Cosine decay: 1 → min_lr_fraction
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = min(progress, 1.0)  # clamp at 1 for safety
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_fraction + (1.0 - min_lr_fraction) * cosine

        super().__init__(optimizer, lr_lambda)


# =============================================================================
# Self-test  (run with: python -m src.training.lr_schedule  from project root)
# =============================================================================

if __name__ == "__main__":
    import torch as _torch
    print("=== experiments.natural_images.common.training.lr_schedule self-test ===\n")

    LR = 1e-3
    total, warmup = 1000, 100
    opt = _torch.optim.SGD([_torch.nn.Parameter(_torch.zeros(1))], lr=LR)
    sched = CosineWarmup(opt, total_steps=total, warmup_steps=warmup)

    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()

    # During warmup: LR increases linearly from 0 to LR
    assert lrs[0] == 0.0, f"LR at step 0 should be 0, got {lrs[0]}"
    assert abs(lrs[warmup - 1] - LR * (warmup - 1) / warmup) < 1e-9
    print(f"[PASS] Warmup: LR at step 0 = {lrs[0]:.2e}, at step {warmup-1} = {lrs[warmup-1]:.6f}")

    # After warmup: LR should be ≈ LR (peak)
    assert abs(lrs[warmup] - LR) < 1e-6, f"Expected peak {LR}, got {lrs[warmup]:.2e}"
    print(f"[PASS] Peak LR at step {warmup}: {lrs[warmup]:.6f}")

    # At end of training: LR should decay to ~0
    assert lrs[-1] < 0.01 * LR, f"LR at end should be near 0, got {lrs[-1]:.2e}"
    print(f"[PASS] Final LR at step {total-1}: {lrs[-1]:.2e}  (near 0)")

    # LR is monotonically non-decreasing during warmup
    assert all(lrs[i] <= lrs[i+1] for i in range(warmup - 1)), "Warmup not monotone"
    print("[PASS] Warmup phase is monotonically increasing")

    # LR is monotonically non-increasing after warmup
    assert all(lrs[i] >= lrs[i+1] for i in range(warmup, total - 1)), "Decay not monotone"
    print("[PASS] Decay phase is monotonically decreasing")

    # min_lr_fraction: end LR should be min_lr_fraction * LR
    opt2 = _torch.optim.SGD([_torch.nn.Parameter(_torch.zeros(1))], lr=LR)
    sched2 = CosineWarmup(opt2, total_steps=200, warmup_steps=10, min_lr_fraction=0.1)
    for _ in range(200):
        sched2.step()
    end_lr = opt2.param_groups[0]["lr"]
    assert abs(end_lr - 0.1 * LR) < 1e-9, f"Expected min_lr={0.1*LR:.2e}, got {end_lr:.2e}"
    print(f"[PASS] min_lr_fraction=0.1 → final LR = {end_lr:.2e}  (= 0.1 × {LR:.0e})")

    print("\n=== All LR schedule tests passed ===")
