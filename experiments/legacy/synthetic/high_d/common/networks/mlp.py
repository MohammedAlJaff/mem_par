"""
Time-conditioned MLP for synthetic experiments.

This is the network used for all synthetic data (GMM, torus, 2D toys).
It's deliberately simple — the closest practical analogue to the random
feature models used in the theory (Bonnaire et al., George et al.).

Architecture:
    Input: [z_t ; time_embed(t)] ∈ R^{D + time_embed_dim}
    Hidden: L layers of (Linear → SiLU) with width W
    Output: Linear → R^D

The time embedding follows the sinusoidal positional encoding from
Vaswani et al. (2017), adapted for continuous t ∈ [0, 1]. This is
now standard practice in diffusion models (DDPM, Score SDE, etc.).

Why sinusoidal and not learned? For continuous time, sinusoidal encoding
provides a smooth, non-degenerate representation of t without requiring
discretization. Learned embeddings would need a discrete time grid.

IMPORTANT: This network outputs a D-dimensional vector. The MEANING of
that vector (x-prediction, ε-prediction, or v-prediction) is determined
by the FlowMatching wrapper, not by this network. The network is agnostic
to parameterization — it just maps (z_t, t) → R^D.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding of scalar time t ∈ [0, 1] → R^{embed_dim}.

    This is the continuous analog of positional encoding. For a scalar t:
        embed[2i]   = sin(t · ω_i)
        embed[2i+1] = cos(t · ω_i)
    where ω_i = 10000^{-2i/embed_dim}.

    The frequencies span several orders of magnitude, giving the network
    access to both coarse and fine time information. Low frequencies capture
    the broad structure (t ≈ 0 vs t ≈ 1), high frequencies capture local
    variation within a time region.

    Why this matters for flow matching:
        The optimal denoising function changes rapidly with t. At t ≈ 0
        (pure noise), the best prediction is the dataset mean. At t ≈ 1
        (nearly clean), the best prediction is the input itself. The time
        embedding must give the network enough information to distinguish
        these regimes. A single scalar t would work in theory but makes
        optimization harder (the network must learn to "switch behavior"
        based on a 1D input).
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar timesteps in [0, 1]
        Returns:
            (B, embed_dim) sinusoidal embeddings
        """
        assert t.dim() == 1, f"Expected 1D tensor, got shape {t.shape}"
        half_dim = self.embed_dim // 2

        # Frequency schedule: log-spaced from 1 to 10000
        # This is identical to Vaswani's formula with max_period=10000
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device) / half_dim
        )

        # Outer product: (B, 1) × (1, half_dim) → (B, half_dim)
        args = t.unsqueeze(1) * freqs.unsqueeze(0)

        # Interleave sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        # Handle odd embed_dim
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(t.unsqueeze(1))], dim=1)

        return embedding


class TimeConditionedMLP(nn.Module):
    """
    Simple MLP conditioned on continuous time via sinusoidal embedding.

    Architecture:
        1. Embed time: t → sinusoidal → (B, time_embed_dim)
        2. Concatenate: [z_t, time_embed] → (B, D + time_embed_dim)
        3. Hidden layers: Linear(in, W) → SiLU → ... → Linear(W, W) → SiLU
        4. Output: Linear(W, D)

    This is the workhorse for synthetic experiments. The key capacity
    parameter is width W (hidden_dim). When W < D, the network has a
    dimensional bottleneck — this is where the Li & He mechanism kicks in.
    When W ≥ D, the network can represent full-rank mappings and the
    parameterization effect should weaken.

    For the paper, we fix W = 256 and vary D ∈ {4, ..., 256}. When D = 256,
    W = D (no bottleneck, control). When D = 256 but d = 4, the bottleneck
    is between W and the target dimensionality, not between W and D.

    Args:
        input_dim: dimension of z_t (= D, the ambient dimension)
        hidden_dim: width of hidden layers (the capacity bottleneck)
        depth: number of hidden layers (3 is usually enough for synthetic)
        time_embed_dim: dimension of sinusoidal time embedding
            64 is standard. Could go lower (32) for very cheap experiments
            or higher (128) if the network struggles with time-dependent behavior.
        activation: 'silu' (default, smooth) or 'relu' (matches RF theory better)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        time_embed_dim: int = 64,
        activation: str = "silu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Choose activation
        # SiLU (Swish) is smooth — better gradients, standard in modern diffusion.
        # ReLU matches the theory (Bonnaire's RF model uses ReLU features).
        # For experiments, use SiLU. For theory validation, consider ReLU.
        if activation == "silu":
            act_fn = nn.SiLU
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        in_features = input_dim + time_embed_dim

        for i in range(depth):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(act_fn())
            in_features = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Initialize output layer to near-zero
        # This means the untrained model outputs ~0, which is a reasonable
        # initialization for flow matching (the mean prediction).
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) noisy input z_t
            t: (B,) continuous timestep in [0, 1]
        Returns:
            (B, D) prediction (meaning depends on pred_space in FlowMatching)
        """
        # Handle t shape: accept (B,) or (B,1)
        if t.dim() == 2:
            t = t.squeeze(1)

        t_emb = self.time_embed(t)          # (B, time_embed_dim)
        h = torch.cat([x, t_emb], dim=1)    # (B, D + time_embed_dim)
        h = self.hidden(h)                   # (B, hidden_dim)
        return self.output_proj(h)           # (B, D)


# =============================================================================
# Self-test  (run with: python -m src.networks.mlp  from project root)
# =============================================================================

if __name__ == "__main__":
    print("=== experiments.synthetic.high_d.common.networks.mlp self-test ===\n")

    B, D = 16, 32

    # --- SinusoidalTimeEmbedding ---
    emb = SinusoidalTimeEmbedding(embed_dim=64)
    t = torch.rand(B)
    out = emb(t)
    assert out.shape == (B, 64), f"Expected (B, 64), got {out.shape}"
    # Nearby t values should give nearby embeddings (smooth)
    t_close = torch.full((B,), 0.5)
    t_far = torch.full((B,), 0.0)
    diff_close = (emb(t_close + 0.001) - emb(t_close)).norm(dim=1).mean()
    diff_far = (emb(t_far + 0.1) - emb(t_far)).norm(dim=1).mean()
    assert diff_close < diff_far, "Embedding not smooth in t"
    print(f"[PASS] SinusoidalTimeEmbedding: output shape {out.shape}, smooth in t")

    # Odd embed_dim should not crash
    emb_odd = SinusoidalTimeEmbedding(embed_dim=33)
    assert emb_odd(t).shape == (B, 33)
    print("[PASS] SinusoidalTimeEmbedding works with odd embed_dim=33")

    # --- TimeConditionedMLP forward pass ---
    for depth in [1, 3]:
        for act in ["silu", "relu"]:
            net = TimeConditionedMLP(input_dim=D, hidden_dim=64, depth=depth,
                                     time_embed_dim=32, activation=act)
            x = torch.randn(B, D)
            t = torch.rand(B)
            out = net(x, t)
            assert out.shape == (B, D), f"Expected ({B}, {D}), got {out.shape}"
    print(f"[PASS] TimeConditionedMLP forward pass → shape ({B}, {D}) for depth∈{{1,3}}, act∈{{silu,relu}}")

    # Output layer is initialised to zero → untrained model outputs ~0
    net0 = TimeConditionedMLP(input_dim=D, hidden_dim=64, depth=2, time_embed_dim=32)
    x0 = torch.randn(B, D)
    t0 = torch.rand(B)
    out0 = net0(x0, t0)
    assert out0.abs().max().item() < 1e-6, f"Output layer not zero-init: max={out0.abs().max():.2e}"
    print(f"[PASS] Zero-init output layer: max abs = {out0.abs().max():.2e}")

    # Accept t of shape (B,) and (B,1)
    t_2d = t0.unsqueeze(1)
    out_2d = net0(x0, t_2d)
    assert out_2d.shape == (B, D)
    print("[PASS] Accepts t of shape (B,) and (B,1)")

    # Parameter count sanity
    n_params = sum(p.numel() for p in net0.parameters())
    print(f"      Parameter count for D={D}, W=64, depth=2: {n_params:,}")

    print("\n=== All MLP tests passed ===")
