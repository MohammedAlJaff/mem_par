"""
Synthetic datasets with controlled intrinsic geometry.

These are the MAIN synthetic experiments (paper §4.1). Unlike the toy2d
distributions, these operate in high-dimensional ambient space with a
known, controlled manifold gap ψ = D/d.

The core experimental logic:
    - d = D (ψ = 1): NULL CONTROL — no manifold gap, no parameterization effect expected
    - d ≪ D (ψ ≫ 1): TEST — manifold gap present, parameterization should matter

Classes:
    GMMSubspace    — K Gaussians on a d-dim linear subspace of R^D
    TorusEmbedding — d-dim torus in R^D via frozen MLP

Both classes produce torch Datasets so they plug directly into DataLoader.

Key design principle: The manifold geometry (basis U, centers μ_k, frozen MLP)
is fixed by the seed and shared across all parameterizations. Only the training
samples vary (controlled by a separate data_seed). This means x-pred, ε-pred,
and v-pred see the SAME data — the only difference is the training target.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional


class GMMSubspace(Dataset):
    """
    K isotropic Gaussians on a d-dimensional linear subspace of R^D.

    Construction (all deterministic given geometry_seed):
        1. Draw random D×d Gaussian matrix, QR decompose → orthonormal U ∈ R^{D×d}
        2. Draw K mode centers μ_k ~ N(0, I_d) in subspace coordinates
        3. For each sample:
            a. Pick mode k uniformly at random
            b. Sample z ~ N(μ_k, σ_z² I_d) in subspace
            c. Project: x = U z + η,  η ~ N(0, σ_η² I_D)

    The off-subspace noise η is physically motivated: real data has some
    variation in all directions, not just the manifold directions. It also
    prevents numerically singular covariance matrices. But it should be
    MUCH smaller than σ_z to preserve the low-rank structure.

    Paper §4.1 spec:
        D = 256, d ∈ {4, 16, 32, 64, 128, 256}, K = 10
        σ_z = 0.5, σ_η = 1e-3
        n ∈ {50, 100, 200, 500, 1000, 2000, 5000, 10000}

    Args:
        D: ambient dimension
        d: intrinsic dimension (subspace rank)
        K: number of mixture components
        sigma_z: intra-mode std in subspace (controls mode width)
            0.1 → point-like modes, easiest to memorize
            0.3 → Li & He's toy setting
            0.5 → moderate spread (our default)
            1.0 → overlapping modes, hardest to tell memorization from generalization
        sigma_eta: off-subspace noise std
            0.0   → exactly on subspace (may cause numerical issues)
            1e-3  → negligible but regularizing (our default)
            0.01  → noticeable off-manifold spread
        n: dataset size
        geometry_seed: controls U and μ_k (fixed across parameterizations)
        data_seed: controls which samples are drawn (fixed across parameterizations)
    """

    def __init__(
        self,
        D: int = 256,
        d: int = 4,
        K: int = 10,
        sigma_z: float = 0.5,
        sigma_eta: float = 1e-3,
        n: int = 1000,
        geometry_seed: int = 0,
        data_seed: int = 0,
    ):
        super().__init__()
        self.D = D
        self.d = d
        self.K = K
        self.sigma_z = sigma_z
        self.sigma_eta = sigma_eta

        # ---- Fixed geometry (deterministic given geometry_seed) ----
        rng_geom = np.random.RandomState(geometry_seed)

        # Orthonormal basis via QR
        random_mat = rng_geom.randn(D, d).astype(np.float32)
        Q, _ = np.linalg.qr(random_mat)
        self.U = torch.tensor(Q[:, :d], dtype=torch.float32)  # (D, d)

        # Mode centers in subspace coordinates
        self.centers = torch.tensor(rng_geom.randn(K, d).astype(np.float32))  # (K, d)

        # ---- Sample data (deterministic given data_seed) ----
        rng_data = np.random.RandomState(data_seed)

        # Assign each point to a mode
        assignments = rng_data.randint(0, K, size=n)

        # Sample in subspace
        z = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            z[i] = rng_data.randn(d) * sigma_z + self.centers[assignments[i]].numpy()
        z = torch.tensor(z)

        # Project to ambient space + off-subspace noise
        x = z @ self.U.T  # (n, D)
        if sigma_eta > 0:
            eta = torch.tensor(rng_data.randn(n, D).astype(np.float32)) * sigma_eta
            x = x + eta

        self.data = x  # (n, D)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    @property
    def intrinsic_dim(self) -> int:
        return self.d

    # NOTE: only GMMSubspace has subspace_basis; TorusEmbedding does not

    @property
    def subspace_basis(self) -> torch.Tensor:
        """(D, d) orthonormal basis. For residual analysis and diagnostics."""
        return self.U


class TorusEmbedding(Dataset):
    """
    d-dimensional torus in R^D via a frozen random MLP.

    Construction:
        1. Sample angles θ ~ Uniform([0, 2π)^d)
        2. Form periodic features: [cos(θ), sin(θ)] ∈ R^{2d}
           (This ensures the map is periodic — the topology is a torus,
            not R^d. Without cos/sin, the map wouldn't "wrap around.")
        3. Map through frozen 2-layer ReLU MLP:
           g(θ) = W₂ · ReLU(W₁ · [cos(θ), sin(θ)] + b₁) + b₂

    Why this matters for the paper:
        The GMM subspace is a LINEAR manifold. The torus is NONLINEAR.
        If the parameterization effect only works with flat manifolds,
        that limits practical relevance. The torus tests robustness
        to curvature.

    Also: the torus is the ONLY setting where we compute FLIPD LID,
    because we know the true intrinsic dimension exactly. This lets
    us measure n*_dim (the dataset size at which the model recovers
    the correct manifold dimension).

    Paper §4.1:
        D = 256, d ∈ {2, 4, 8}

    Args:
        D: ambient dimension
        d: intrinsic dimension (torus dimension)
        hidden_mult: MLP width multiplier (width = hidden_mult × D)
            2 → constrained surface, may self-intersect
            4 → smooth, well-conditioned (default)
            8 → very smooth, more degrees of freedom
        n: dataset size
        geometry_seed: controls frozen MLP weights
        data_seed: controls which angles are sampled
    """

    def __init__(
        self,
        D: int = 256,
        d: int = 2,
        hidden_mult: int = 4,
        n: int = 1000,
        geometry_seed: int = 0,
        data_seed: int = 0,
    ):
        super().__init__()
        self.D = D
        self.d = d
        h = hidden_mult * D

        # ---- Frozen MLP weights (deterministic given geometry_seed) ----
        torch.manual_seed(geometry_seed)
        self.W1 = torch.randn(h, 2 * d) / (2 * d) ** 0.5
        self.b1 = torch.randn(h) * 0.1
        self.W2 = torch.randn(D, h) / h**0.5
        self.b2 = torch.randn(D) * 0.1

        # ---- Sample data ----
        torch.manual_seed(data_seed)
        theta = 2 * np.pi * torch.rand(n, d)

        # Periodic features — this is what makes it a torus
        # cos/sin ensure that θ = 0 and θ = 2π map to the same point
        features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)  # (n, 2d)

        # Forward through frozen MLP
        with torch.no_grad():
            hidden = torch.nn.functional.leaky_relu(
                features @ self.W1.T + self.b1, negative_slope=0.01
            )  # (n, h)
            x = hidden @ self.W2.T + self.b2  # (n, D)

        self.data = x

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    @property
    def intrinsic_dim(self) -> int:
        return self.d


# =============================================================================
# Self-test  (run with: python -m src.data.synthetic  from project root)
# =============================================================================

if __name__ == "__main__":
    print("=== experiments.synthetic.high_d.common.data.synthetic self-test ===\n")

    # --- GMMSubspace ---
    D, d, K, n = 64, 4, 5, 200
    ds = GMMSubspace(D=D, d=d, K=K, sigma_z=0.5, sigma_eta=1e-3, n=n, geometry_seed=0, data_seed=1)

    assert len(ds) == n
    assert ds[0].shape == (D,)
    print(f"[PASS] GMMSubspace: len={len(ds)}, item shape={tuple(ds[0].shape)}")

    # Basis should be orthonormal
    U = ds.subspace_basis  # (D, d)
    UtU = U.T @ U
    assert torch.allclose(UtU, torch.eye(d), atol=1e-5), "Basis not orthonormal"
    print(f"[PASS] GMMSubspace basis is orthonormal  U^T·U ≈ I_{d}")

    # geometry_seed is decoupled from data_seed
    ds_other = GMMSubspace(D=D, d=d, K=K, n=n, geometry_seed=0, data_seed=99)
    assert torch.allclose(ds.subspace_basis, ds_other.subspace_basis)
    print("[PASS] geometry_seed is decoupled from data_seed")

    # off-subspace residual should be tiny (sigma_eta=1e-3)
    proj = ds.data @ U
    x_rec = proj @ U.T
    residual = (ds.data - x_rec).norm(dim=1)
    assert residual.mean() < 1e-3 * D**0.5 * 5
    print(f"[PASS] off-subspace residual mean={residual.mean():.5f}  (σ_η=1e-3)")

    # --- TorusEmbedding ---
    D_t, d_t, n_t = 32, 2, 150
    tds = TorusEmbedding(D=D_t, d=d_t, hidden_mult=4, n=n_t, geometry_seed=0, data_seed=0)
    assert len(tds) == n_t
    assert tds[0].shape == (D_t,)
    print(f"\n[PASS] TorusEmbedding: len={len(tds)}, item shape={tuple(tds[0].shape)}, intrinsic_dim={tds.intrinsic_dim}")

    tds2 = TorusEmbedding(D=D_t, d=d_t, n=n_t, geometry_seed=0, data_seed=99)
    assert torch.allclose(tds.W1, tds2.W1), "geometry_seed should fix MLP weights"
    assert not torch.allclose(tds.data, tds2.data), "data_seed should vary data"
    print("[PASS] TorusEmbedding geometry_seed ↔ data_seed separation works")

    print("\n=== All synthetic data tests passed ===")
