"""
sphere_data.py
==============

Synthetic hypersphere-shell data for intrinsic-vs-ambient memorization
experiments.

Construction
------------
Ambient space:                     x in R^D
Intrinsic manifold dimension:     K
Sphere support:                   S^K embedded in a random (K + 1)-dimensional
                                  subspace of R^D

Clean shell sample:
    u ~ Uniform(S^K) in R^(K+1)
    x_shell = radius * u @ B

Here B in R^((K+1) x D) has orthonormal rows, so A = B^T B is the orthogonal
projector onto the sphere-supporting subspace.

Observed sample:
    x = x_shell + off_manifold_sigma * z_perp
    z_perp = xi (I - A),   xi ~ N(0, I_D)

So the clean geometry is a sphere shell, while thickness is added only in the
orthogonal complement.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

PARAMS = ("x", "eps", "v")


class HypersphereShellDataset:
    """Thin dataset class for a random embedded hypersphere shell."""

    def __init__(
        self,
        d: int,
        k: int,
        radius: float = 1.0,
        off_manifold_sigma: float = 0.05,
        n_train: int = 2000,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.d = int(d)
        self.k = int(k)
        self.subspace_dim = self.k + 1
        self.radius = float(radius)
        self.off_manifold_sigma = float(off_manifold_sigma)
        self.n_train = int(n_train)

        if self.d < 1:
            raise ValueError("d must be at least 1.")
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        if self.subspace_dim > self.d:
            raise ValueError("need k + 1 <= d for an S^K shell in R^d.")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")
        if self.off_manifold_sigma < 0.0:
            raise ValueError("off_manifold_sigma must be non-negative.")
        if self.n_train < 1:
            raise ValueError("n_train must be at least 1.")

        g = torch.Generator(device=self.device).manual_seed(seed)
        G = torch.randn(self.d, self.d, generator=g, device=self.device, dtype=dtype)
        Qfull, _ = torch.linalg.qr(G)
        B_full = Qfull.T
        self.B = B_full[: self.subspace_dim].contiguous()
        self.A = self.B.T @ self.B
        self._I_d = torch.eye(self.d, device=self.device, dtype=dtype)
        self.A_perp = self._I_d - self.A

        eye_subspace = torch.eye(self.subspace_dim, device=self.device, dtype=dtype)
        assert torch.allclose(self.B @ self.B.T, eye_subspace, atol=1e-8), (
            "B rows not orthonormal"
        )
        assert torch.allclose(self.A, self.A.T, atol=1e-8), "A not symmetric"
        assert torch.allclose(self.A @ self.A, self.A, atol=1e-6), "A not idempotent"
        assert abs(self.A.diagonal().sum().item() - self.subspace_dim) < 1e-4, (
            "rank(A) != k + 1"
        )

        self._gen_train = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.train_set = self._sample(self.n_train, self._gen_train)

    def _sample_shell(self, n: int, gen: torch.Generator) -> Tensor:
        coords = torch.randn(
            n,
            self.subspace_dim,
            generator=gen,
            device=self.device,
            dtype=self.dtype,
        )
        norms = torch.linalg.norm(coords, dim=-1, keepdim=True).clamp_min(1.0e-12)
        unit = coords / norms
        return self.radius * (unit @ self.B)

    def _sample(self, n: int, gen: torch.Generator) -> Tensor:
        x_shell = self._sample_shell(n, gen)
        xi = torch.randn(
            n, self.d, generator=gen, device=self.device, dtype=self.dtype
        )
        z_perp = xi @ self.A_perp
        return x_shell + self.off_manifold_sigma * z_perp

    def sample(self, n: int, seed: int | None = None) -> Tensor:
        gen = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        if gen is None:
            gen = torch.Generator(device=self.device)
            gen.seed()
        return self._sample(int(n), gen)

    def project_to_subspace(self, x: Tensor) -> Tensor:
        x = x.to(self.device, self.dtype)
        return x @ self.A

    def subspace_coordinates(self, x: Tensor) -> Tensor:
        x = x.to(self.device, self.dtype)
        return x @ self.B.T

    def dist_to_manifold(self, x: Tensor) -> Tensor:
        x = x.to(self.device, self.dtype)
        return torch.linalg.norm(x - x @ self.A, dim=-1)

    def projected_radius_error(self, x: Tensor) -> Tensor:
        coords = self.subspace_coordinates(x)
        radii = torch.linalg.norm(coords, dim=-1)
        return (radii - self.radius).abs()

    @staticmethod
    def f_mem(r: Tensor, rho: float) -> Tensor:
        return (r < rho).to(r.dtype).mean()

    def __repr__(self):
        return (
            f"HypersphereShellDataset(d={self.d}, k={self.k}, "
            f"subspace_dim={self.subspace_dim}, radius={self.radius:g}, "
            f"off_manifold_sigma={self.off_manifold_sigma:g}, "
            f"n_train={self.n_train}, dtype={self.dtype})"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    ds = HypersphereShellDataset(
        d=3,
        k=2,
        radius=1.5,
        off_manifold_sigma=0.1,
        n_train=512,
        seed=0,
    )
    print(ds)

    xb = ds.sample(4096, seed=123)
    projected_radii = torch.linalg.norm(ds.subspace_coordinates(xb), dim=-1)
    print(
        "[1] projected radius mean = %.4f, target radius = %.4f"
        % (projected_radii.mean().item(), ds.radius)
    )

    md = ds.dist_to_manifold(xb)
    expected = ds.off_manifold_sigma * math.sqrt(max(ds.d - ds.subspace_dim, 0))
    print(
        "[2] mean off-manifold dist = %.4f, expected scale = %.4f"
        % (md.mean().item(), expected)
    )

    radius_mae = ds.projected_radius_error(xb).mean().item()
    print("[3] projected radius MAE = %.4e" % radius_mae)
