"""
Lab 03 — Network blocks: TimeConditionedMLP (and optionally SmallUNet).

What this covers:
    - TimeConditionedMLP: architecture, parameter counts, forward pass
    - SinusoidalTimeEmbedding: what it produces
    - The network is parameterization-AGNOSTIC — it just maps (z_t, t) → R^D
    - Capacity: W=256 vs D (the bottleneck regime relevant to the paper)
    - (Optional) SmallUNet forward pass if you want to see the image-model

Run from project root:
    python lab/03_network_blocks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from experiments.synthetic.high_d.common.networks import (
    SinusoidalTimeEmbedding,
    TimeConditionedMLP,
)
from experiments.synthetic.high_d.common.utils import seed_everything


def demo_sinusoidal_embedding():
    print("\n── 1. SinusoidalTimeEmbedding ────────────────────────────────────")
    emb = SinusoidalTimeEmbedding(embed_dim=64)

    # Show embeddings at a few t values
    print("  Embeddings at t ∈ {0.0, 0.25, 0.5, 0.75, 1.0}:")
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t_tensor = torch.tensor([t_val])
        e = emb(t_tensor)
        print(f"  t={t_val:.2f}  →  shape={tuple(e.shape)}, "
              f"min={e.min().item():.3f}, max={e.max().item():.3f}, "
              f"norm={e.norm().item():.3f}")

    # Embedding is smooth in t (nearby t → nearby embedding)
    t_range = torch.linspace(0.01, 0.99, 100)
    embeddings = emb(t_range)  # (100, 64)
    diffs = (embeddings[1:] - embeddings[:-1]).norm(dim=1)
    print(f"\n  Consecutive-t embedding diffs: mean={diffs.mean():.4f}, max={diffs.max():.4f}")
    print("  (smooth → network can interpolate across time)")


def demo_mlp_architecture():
    print("\n── 2. TimeConditionedMLP: architecture ───────────────────────────")
    D, W, L, T = 256, 256, 3, 64
    net = TimeConditionedMLP(input_dim=D, hidden_dim=W, depth=L, time_embed_dim=T)

    print(f"  Input dim D  = {D}")
    print(f"  Hidden dim W = {W}  ← capacity bottleneck")
    print(f"  Depth        = {L}")
    print(f"  Time embed   = {T}")
    print(f"\n  Architecture:")
    print(f"    (z_t: D) + (time_embed: T) → concat → (D+T,)")
    print(f"    Linear({D+T}, {W}) → SiLU   [layer 0]")
    for i in range(1, L):
        print(f"    Linear({W}, {W}) → SiLU   [layer {i}]")
    print(f"    Linear({W}, {D})             [output, zero-init]")

    total_params = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\n  Total params   : {total_params:,}")
    print(f"  Trainable      : {trainable:,}")


def demo_mlp_forward():
    print("\n── 3. TimeConditionedMLP: forward pass ───────────────────────────")
    seed_everything(0)
    B, D = 32, 256
    net = TimeConditionedMLP(input_dim=D, hidden_dim=256, depth=3, time_embed_dim=64)
    net.eval()

    z_t = torch.randn(B, D)
    t   = torch.rand(B)

    with torch.no_grad():
        out = net(z_t, t)

    print(f"  Input  z_t: shape {tuple(z_t.shape)}")
    print(f"  Input  t:   shape {tuple(t.shape)}")
    print(f"  Output:     shape {tuple(out.shape)}")
    print(f"  Output max abs  = {out.abs().max().item():.6f}  (zero-init → ≈ 0 before training)")

    # Shape is preserved: output matches input D
    assert out.shape == (B, D)
    print("  ✓ output shape matches (B, D)")


def demo_mlp_capacity_sweep():
    print("\n── 4. Capacity sweep: W vs D (the bottleneck regime) ────────────")
    print("  In the paper: fixed W=256, varying D.")
    print("  When D < W: network has EXCESS capacity (easy to memorize).")
    print("  When D = W: balanced (our control condition ψ=1).")
    print("  When D > W: bottleneck — but D is fixed at 256 in our setup.")
    print()
    B = 16
    fixed_W = 256
    print(f"  {'D':>6}  {'W':>6}  {'W/D':>6}  {'params':>10}  {'out shape'}")
    for D in [4, 16, 32, 64, 128, 256]:
        net = TimeConditionedMLP(input_dim=D, hidden_dim=fixed_W, depth=3, time_embed_dim=64)
        n_p = sum(p.numel() for p in net.parameters())
        z = torch.randn(B, D)
        t = torch.rand(B)
        out = net(z, t)
        print(f"  {D:>6}  {fixed_W:>6}  {fixed_W/D:>6.1f}  {n_p:>10,}  {tuple(out.shape)}")


def demo_mlp_activations():
    print("\n── 5. SiLU vs ReLU activations ───────────────────────────────────")
    D, B = 32, 8
    seed_everything(0)
    net_silu = TimeConditionedMLP(input_dim=D, hidden_dim=64, depth=2, activation="silu")
    seed_everything(0)
    net_relu = TimeConditionedMLP(input_dim=D, hidden_dim=64, depth=2, activation="relu")

    # Train for a few steps to see non-zero outputs
    opt_s = torch.optim.Adam(net_silu.parameters(), lr=1e-3)
    opt_r = torch.optim.Adam(net_relu.parameters(), lr=1e-3)
    z = torch.randn(B, D)
    t = torch.rand(B)
    target = torch.randn(B, D)

    for _ in range(50):
        loss_s = ((net_silu(z, t) - target) ** 2).mean()
        opt_s.zero_grad(); loss_s.backward(); opt_s.step()
        loss_r = ((net_relu(z, t) - target) ** 2).mean()
        opt_r.zero_grad(); loss_r.backward(); opt_r.step()

    out_s = net_silu(z, t).detach()
    out_r = net_relu(z, t).detach()
    print(f"  After 50 training steps on random target:")
    print(f"  SiLU: loss={loss_s.item():.4f}, out range=[{out_s.min():.3f}, {out_s.max():.3f}]")
    print(f"  ReLU: loss={loss_r.item():.4f}, out range=[{out_r.min():.3f}, {out_r.max():.3f}]")
    print("  SiLU is smooth → better gradients. ReLU matches theory (Bonnaire RF model).")


if __name__ == "__main__":
    print("=" * 65)
    print("  Lab 03: Network blocks (TimeConditionedMLP)")
    print("=" * 65)

    demo_sinusoidal_embedding()
    demo_mlp_architecture()
    demo_mlp_forward()
    demo_mlp_capacity_sweep()
    demo_mlp_activations()

    print("\n" + "=" * 65)
    print("  Done. Network blocks understood.")
    print("=" * 65)
