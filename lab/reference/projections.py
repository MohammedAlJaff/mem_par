"""
Projections: embed low-dimensional data into high-dimensional ambient space.

These are the "knobs" that control ψ = D/d in synthetic experiments.
The key idea from Li & He (2025): when d ≪ D, x-prediction has an advantage
because the target lies on a d-dimensional manifold, while ε-prediction
targets the full D-dimensional ambient space.

Two projection types:
    OrthogonalProjection — linear subspace (distances exactly preserved)
    RandomMLPProjection  — curved manifold (more realistic, distances distorted)

Both are FROZEN after construction. The projection is part of the data
generation process, not the learning process. All parameterizations see
the same projected data.

Pedagogical note:
    Why orthogonal? The QR decomposition of a random Gaussian matrix gives
    a uniformly random d-dimensional subspace of R^D. By construction,
    P^T P = I_d, so Euclidean distances in the subspace are preserved in
    ambient space. This is the cleanest possible manifold: zero curvature,
    known dimension, exact distance preservation. Any difference between
    parameterizations is purely from the ambient dimension gap.

    Why random MLP? Real manifolds are curved. A frozen 2-layer ReLU MLP
    maps R^d → R^D through a nonlinear, non-isometric embedding. The
    manifold has curvature, the metric is distorted, and the Jacobian
    varies across the surface. This is much closer to how real image
    manifolds sit inside pixel space.
"""

import torch
import numpy as np


class OrthogonalProjection:
    """
    Project from R^d to R^D via a random orthonormal matrix P ∈ R^{D×d}.

    P is constructed by QR decomposition of a random Gaussian matrix.
    This gives a uniformly random d-dimensional subspace of R^D.

    Properties:
        - P^T P = I_d  (isometric: distances preserved exactly)
        - rank(P) = d
        - The manifold is a flat d-dimensional plane in R^D

    This is the projection used in Li & He (2025) Figure 2 and the
    cleanest setting for testing the ψ = D/d mechanism.

    Args:
        D: ambient dimension (e.g., 256)
        d: intrinsic dimension (e.g., 4)
        seed: controls which random subspace is chosen
    """

    def __init__(self, D: int, d: int, seed: int = 0):
        assert d <= D, f"Intrinsic dim d={d} must be ≤ ambient dim D={D}"
        self.D = D
        self.d = d

        # Generate orthonormal basis via QR decomposition
        # --------------------------------------------------
        # Step 1: Draw a random D×d Gaussian matrix
        # Step 2: QR decompose it → Q is D×D orthogonal, R is D×d upper tri
        # Step 3: Take first d columns of Q → orthonormal basis for a random subspace
        #
        # This is equivalent to sampling uniformly from the Stiefel manifold V_{d,D},
        # which is the set of all d-frames in R^D. A standard result in random matrix
        # theory (see e.g., Mezzadri 2007).
        rng = np.random.RandomState(seed)
        random_mat = rng.randn(D, d).astype(np.float32)
        Q, _ = np.linalg.qr(random_mat)
        # Q is D×d with orthonormal columns (since d ≤ D, QR gives a "thin" Q)
        # Actually np.linalg.qr returns D×D by default. We take first d columns.
        self.P = torch.tensor(Q[:, :d], dtype=torch.float32)  # (D, d)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map from intrinsic coordinates z ∈ R^d to ambient space R^D.

        Args:
            z: (n, d) points in intrinsic coordinates
        Returns:
            x: (n, D) points in ambient space

        Since P^T P = I_d, we have: ||Pz₁ - Pz₂||² = ||z₁ - z₂||²
        (distances are exactly preserved).
        """
        return z @ self.P.T  # (n, d) @ (d, D) = (n, D)

    def project_back(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project from ambient space back to intrinsic coordinates.
        x_intrinsic = P^T x. Only exact if x lies on the subspace.
        Useful for diagnostics (e.g., checking residuals off-manifold).
        """
        return x @ self.P  # (n, D) @ (D, d) = (n, d)

    def residual_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ||x - P P^T x||, the off-subspace component.
        Should be ~0 for data on the manifold, ~σ_η for data with noise.
        """
        x_on = self.project(self.project_back(x))
        return torch.norm(x - x_on, dim=1)


class RandomMLPProjection:
    """
    Project from R^d to R^D via a frozen 2-layer ReLU MLP.

    g(z) = W₂ · ReLU(W₁ z + b₁) + b₂

    This creates a CURVED d-dimensional manifold in R^D. Unlike the
    orthogonal projection, distances are not preserved, the Jacobian
    varies across the surface, and the manifold has nonzero curvature.

    This is more realistic: real data manifolds (images, molecules, etc.)
    are curved surfaces, not flat planes. If the parameterization effect
    only appears with flat manifolds, that would limit its practical relevance.

    Args:
        D: ambient dimension
        d: intrinsic dimension
        hidden_mult: MLP width = hidden_mult × D
            Default 4 gives a smooth, well-conditioned surface.
            Smaller values (2) give a more constrained, potentially
            self-intersecting surface. Larger values (8) give a
            smoother embedding with more degrees of freedom.
        seed: controls the frozen weights
    """

    def __init__(self, D: int, d: int, hidden_mult: int = 4, seed: int = 0):
        self.D = D
        self.d = d
        h = hidden_mult * D

        # Freeze weights at construction time
        # The scaling (1/sqrt(fan_in)) is Xavier-like — ensures the output
        # has reasonable magnitude regardless of d and D.
        torch.manual_seed(seed)
        self.W1 = torch.randn(h, d) / (d**0.5)
        self.b1 = torch.randn(h) * 0.1
        self.W2 = torch.randn(D, h) / (h**0.5)
        self.b2 = torch.randn(D) * 0.1

    @torch.no_grad()
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map from intrinsic coordinates z ∈ R^d to ambient space R^D.

        Args:
            z: (n, d) points in intrinsic coordinates
        Returns:
            x: (n, D) points on the curved manifold

        Note: this is NOT distance-preserving. Two points that are close
        in intrinsic coordinates may be mapped to varying distances in
        ambient space, depending on the local Jacobian.
        """
        h = torch.relu(z @ self.W1.T + self.b1)  # (n, hidden)
        x = h @ self.W2.T + self.b2  # (n, D)
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test the RandomMLPProjection
    D = 10
    d = 2
    projection = RandomMLPProjection(D, d, hidden_mult=4, seed=0)

    # Generate some test points in intrinsic coordinates
    z = torch.randn(5, d)

    # Project to ambient space
    x = projection.project(z)

    print(f"Input shape (intrinsic): {z.shape}")
    print(f"Output shape (ambient): {x.shape}")
    print(f"First point in ambient space: {x[0]}")

    # Plot the first two dimensions of the projected points
    plt.scatter(x[:, 0], x[:, 1])
    plt.title("Projected points in ambient space (first two dimensions)")
    plt.show()
