"""
2D toy distributions for sanity checks and Stage 1 controls.

These operate in R^2 (d = D = 2), which is the NULL CONTROL setting:
when there is no manifold gap (ψ = D/d = 1), the parameterization
hypothesis predicts NO difference between x/ε/v-prediction.

If we see a difference here, something is wrong (or something beyond
the manifold argument is happening — which would be interesting but
would require a different explanation).

Available distributions:
    8-Gaussians — 8 isotropic modes on a circle (good for mode coverage)
    Annulus     — uniform on a ring (good for LID estimation)
    Two moons   — non-convex, connected support (stress test)
"""

import torch
import numpy as np
from typing import Optional


def sample_8gaussians(
    n: int,
    sigma: float = 0.1,  # HYPERPARAMETER: intra-mode spread
    # 0.05 → tight clusters, easy to separate
    # 0.1  → moderate spread (default)
    # 0.3  → overlapping modes, harder
    radius: float = 2.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    8 isotropic Gaussians equally spaced on a circle of given radius.

    This is the standard toy distribution for generative model sanity checks.
    The 8-mode structure makes it easy to visually assess:
      - Mode coverage: does the model find all 8 modes?
      - Mode quality: are generated points clustered around true centers?
      - Memorization: are generated points sitting exactly on training points?

    Args:
        n: number of samples
        sigma: standard deviation of each Gaussian mode
        radius: radius of the circle on which mode centers sit
        seed: if given, makes sampling deterministic

    Returns:
        (n, 2) tensor of samples
    """
    if seed is not None:
        torch.manual_seed(seed)

    centers = get_8gaussian_centers(radius)  # (8, 2)
    assignments = torch.randint(0, 8, (n,))
    samples = centers[assignments] + sigma * torch.randn(n, 2)
    return samples


def get_8gaussian_centers(radius: float = 2.0) -> torch.Tensor:
    """
    Return the 8 mode centers. Useful for:
      - Computing mode coverage metric
      - Plotting ground truth
      - Checking nearest-neighbor assignments
    """
    angles = torch.linspace(0, 2 * np.pi, 9)[:-1]  # 8 evenly spaced angles
    centers = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=1
    )
    return centers  # (8, 2)


def sample_annulus(
    n: int,
    r_inner: float = 1.0,
    r_outer: float = 2.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Uniform samples on an annulus (ring).

    The annulus has intrinsic dimension 2 (it's a 2D region) but its
    support has a topological hole. This makes it useful for testing
    whether the model correctly captures the hole (doesn't generate
    points in the center).

    Also useful for LID estimation: the true local intrinsic dimension
    is 2 everywhere except near the boundaries (where it's ~1).

    Sampling uses rejection: sample in the bounding box, keep points
    in the annulus. Efficient when r_inner is not too close to r_outer.

    Args:
        n: number of samples
        r_inner: inner radius
        r_outer: outer radius
        seed: optional deterministic seed

    Returns:
        (n, 2) tensor
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Rejection sampling in polar coordinates
    # To get uniform in area, sample r² uniformly in [r_inner², r_outer²]
    # (because area element dA = r dr dθ, so P(r) ∝ r → sample r² uniformly)
    r_sq = torch.empty(n).uniform_(r_inner**2, r_outer**2)
    r = torch.sqrt(r_sq)
    theta = torch.empty(n).uniform_(0, 2 * np.pi)

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)


def sample_two_moons(
    n: int,
    noise: float = 0.05,  # HYPERPARAMETER: Gaussian noise around the arcs
    # 0.01 → very thin moons
    # 0.05 → moderate (default)
    # 0.15 → thick, almost overlapping
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Two interleaving half-circles (moons).

    This distribution is non-convex and has two connected components
    (roughly). It's a stress test for the sampler: a model must learn
    to avoid the gap between the two moons.

    The intrinsic dimension is ~1 (each moon is roughly a 1D arc
    fattened by noise), embedded in R^2.

    Args:
        n: number of samples
        noise: Gaussian noise std perpendicular to the arc
        seed: optional seed

    Returns:
        (n, 2) tensor
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Upper moon: semicircle centered at (0, 0)
    n_upper = n // 2
    n_lower = n - n_upper

    theta_upper = torch.empty(n_upper).uniform_(0, np.pi)
    x_upper = torch.cos(theta_upper)
    y_upper = torch.sin(theta_upper)

    # Lower moon: semicircle centered at (1, -0.25), flipped
    theta_lower = torch.empty(n_lower).uniform_(0, np.pi)
    x_lower = 1.0 - torch.cos(theta_lower)
    y_lower = -0.25 - torch.sin(theta_lower)

    x = torch.cat([x_upper, x_lower])
    y = torch.cat([y_upper, y_lower])

    # Add noise
    x += noise * torch.randn_like(x)
    y += noise * torch.randn_like(y)

    return torch.stack([x, y], dim=1)
