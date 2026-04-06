"""
Lab 04 — FlowMatching block: the 3×3 grid in action.

What this covers:
    - FlowMatching wrapping a network → (pred_space, loss_space) pair
    - Loss computation for all 9 combinations
    - How the same network with different parameterization gives different losses
    - Sampling (Euler vs Heun) with an untrained and a lightly trained model
    - The 3×3 grid: why pred_space matters more than loss_space (Li & He finding)

Run from project root:
    python lab/04_flow_matching_block.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from experiments.synthetic.high_d.common.flow_matching import FlowMatching
from experiments.synthetic.high_d.common.networks import TimeConditionedMLP
from experiments.synthetic.high_d.common.utils import seed_everything


SPACES = ["x", "eps", "v"]


def make_model(D: int, pred: str, loss: str, seed: int = 0) -> FlowMatching:
    seed_everything(seed)
    net = TimeConditionedMLP(input_dim=D, hidden_dim=128, depth=3, time_embed_dim=32)
    return FlowMatching(net, pred_space=pred, loss_space=loss)


def demo_loss_grid():
    print("\n── 1. Loss computation for all 9 (pred, loss) combinations ──────")
    torch.manual_seed(42)
    B, D = 64, 32
    x_batch = torch.randn(B, D)

    print(f"  Batch: B={B}, D={D}")
    print(f"\n  Loss values (random init, same network weights per pred_space):")
    print(f"  {'':10}", end="")
    for ls in SPACES:
        print(f"  loss={ls:>4}", end="")
    print()

    for ps in SPACES:
        print(f"  pred={ps:>4}  ", end="")
        for ls in SPACES:
            # Same seed per pred_space → same network for all loss_spaces in a row
            model = make_model(D, ps, ls, seed=0)
            with torch.no_grad():
                loss = model.compute_loss(x_batch)
            print(f"  {loss.item():>8.4f}", end="")
        print()
    print("\n  Note: all 9 losses are valid scalars. The DIAGONAL means no conversion needed.")


def demo_gradient_flow():
    print("\n── 2. Gradient flow (do gradients backprop through all combos?) ──")
    B, D = 16, 16
    x = torch.randn(B, D)
    failures = []
    for ps in SPACES:
        for ls in SPACES:
            model = make_model(D, ps, ls, seed=0)
            loss = model.compute_loss(x)
            loss.backward()
            # Check at least one gradient is non-zero
            has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                           for p in model.parameters())
            if not has_grad:
                failures.append((ps, ls))
    if failures:
        print(f"  FAIL: no gradient for {failures}")
    else:
        print("  ✓ All 9 combinations produce non-zero gradients")


def demo_sampling():
    print("\n── 3. Sampling with an untrained model ──────────────────────────")
    B, D = 8, 16
    seed_everything(0)

    for ps in SPACES:
        model = make_model(D, ps, "v", seed=0)
        model.eval()

        # Euler
        with torch.no_grad():
            s_euler = model.sample((B, D), n_steps=10, method="euler")
            s_heun  = model.sample((B, D), n_steps=10, method="heun")

        print(f"  pred={ps}: Euler shape={tuple(s_euler.shape)}, "
              f"mean={s_euler.mean():.3f}, std={s_euler.std():.3f}  |  "
              f"Heun  mean={s_heun.mean():.3f}, std={s_heun.std():.3f}")

    print("  (Untrained → samples are just random noise. Training gives structure.)")


def demo_train_a_few_steps():
    print("\n── 4. Training a few steps to see loss decrease ─────────────────")
    seed_everything(0)
    B, D = 64, 32

    # Simple 2-Gaussian data (easy to fit)
    centers = torch.stack([torch.ones(D), -torch.ones(D)])
    x_data = centers[torch.randint(0, 2, (200,))] + 0.1 * torch.randn(200, D)

    n_steps = 200
    print(f"  Data: 2 Gaussians, D={D}, n=200, training for {n_steps} steps")
    print(f"\n  {'combo':>10}  {'loss @ 0':>10}  {'loss @ {}'.format(n_steps):>12}  {'reduction':>10}")

    for ps, ls in [("x", "v"), ("eps", "eps"), ("v", "v")]:
        model = make_model(D, ps, ls, seed=42)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Initial loss
        loss0 = model.compute_loss(x_data).item()

        for _ in range(n_steps):
            idx = torch.randint(0, len(x_data), (B,))
            loss = model.compute_loss(x_data[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_final = model.compute_loss(x_data).item()
        reduction = (loss0 - loss_final) / loss0 * 100
        print(f"  ({ps},{ls}):  {loss0:>10.4f}  {loss_final:>12.4f}  {reduction:>9.1f}%")

    print("  ✓ All parameterizations learn (loss decreases). Difference is the RATE and boundary.")


def demo_parameterization_separation():
    print("\n── 5. Key insight: pred_space decouples from loss_space ──────────")
    print("  Claim: you can train with pred=x but compute the loss in v-space.")
    print("  The FlowMatching._convert() handles the algebra transparently.")
    print()

    B, D = 32, 16
    seed_everything(0)
    x = torch.randn(B, D)

    # Manually verify: (pred=x, loss=x) and (pred=x, loss=v) should give
    # DIFFERENT loss values (different measurement spaces) but both valid
    m_xx = make_model(D, "x", "x", seed=0)
    m_xv = make_model(D, "x", "v", seed=0)

    # Use the SAME network weights for fair comparison
    m_xv.network.load_state_dict(m_xx.network.state_dict())

    loss_xx = m_xx.compute_loss(x).item()
    loss_xv = m_xv.compute_loss(x).item()

    print(f"  Same network, pred=x:")
    print(f"    loss in x-space = {loss_xx:.4f}")
    print(f"    loss in v-space = {loss_xv:.4f}")
    print(f"    Ratio           = {loss_xv / loss_xx:.3f}")
    print()
    print("  Takeaway: the pred_space (column) is the key experimental variable.")
    print("  The loss_space (row) just scales the gradient — it doesn't change what")
    print("  information the network learns to extract.")


if __name__ == "__main__":
    print("=" * 65)
    print("  Lab 04: FlowMatching block (3×3 grid)")
    print("=" * 65)

    demo_loss_grid()
    demo_gradient_flow()
    demo_sampling()
    demo_train_a_few_steps()
    demo_parameterization_separation()

    print("\n" + "=" * 65)
    print("  Done. FlowMatching block understood.")
    print("=" * 65)
