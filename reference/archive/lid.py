"""
Local Intrinsic Dimension (LID) estimation via FLIPD.

This is used ONLY for the torus experiments where the true intrinsic
dimension d is known exactly. It measures n*_dim: the dataset size at
which the model recovers the correct manifold dimension.

n*_dim is distinct from n*_mem:
    n*_mem: when the model stops COPYING training points
    n*_dim: when the model recovers the correct DIMENSIONALITY

A model can stop copying (n > n*_mem) while still generating samples
on a lower-dimensional subspace of the true manifold (n < n*_dim).
This is the "geometric memorization" regime discussed in the paper.

FLIPD (Fast Local Intrinsic Dimensionality via Pairwise Distances)
is a computationally efficient LID estimator. It works by analyzing
the scaling of pairwise distances in local neighborhoods.

STATUS: Stub implementation. For the actual FLIPD algorithm, either:
    1. Use scikit-dimension: pip install scikit-dimension
       from skdim.id import FisherS  (or similar)
    2. Implement from the paper: [TODO: add FLIPD citation]

This module is LOW PRIORITY. Only needed if torus experiments show
an interesting gap between n*_mem and n*_dim.
"""

import torch
import numpy as np
from typing import Optional


def estimate_lid(
    samples: torch.Tensor,
    k: int = 20,
    method: str = "mle",
) -> float:
    """
    Estimate the average local intrinsic dimension of a point cloud.

    Args:
        samples: (M, D) generated samples
        k: number of nearest neighbors for local estimate
        method: estimation method
            'mle': Maximum Likelihood Estimator (Levina & Bickel 2004)
                Simple, fast, well-understood. Tends to underestimate.
            'flipd': FLIPD estimator (TODO: implement)
                More robust but more complex.

    Returns:
        Estimated average LID (scalar)
    """
    if method == "mle":
        return _lid_mle(samples.numpy(), k)
    elif method == "flipd":
        raise NotImplementedError(
            "FLIPD not yet implemented. Use method='mle' or install scikit-dimension:\n"
            "  pip install scikit-dimension\n"
            "  from skdim.id import MLE\n"
            "  lid = MLE().fit_transform(samples.numpy())"
        )
    else:
        raise ValueError(f"Unknown LID method: {method}")


def _lid_mle(X: np.ndarray, k: int = 20) -> float:
    """
    Maximum Likelihood Estimator for local intrinsic dimension.

    For each point, the MLE estimate is:
        d_hat(x) = -k / sum_{j=1}^{k} log(||x - x_j|| / ||x - x_k||)

    where x_1, ..., x_k are the k nearest neighbors of x, ordered by distance.

    The global estimate is the average over all points.

    Reference: Levina & Bickel (2004) "Maximum Likelihood Estimation of
    Intrinsic Dimension"
    """
    from scipy.spatial import KDTree

    n, d = X.shape
    tree = KDTree(X)

    # Query k+1 neighbors (first neighbor is the point itself)
    dists, _ = tree.query(X, k=k + 1)
    dists = dists[:, 1:]  # remove self-distance (column 0)

    # Avoid log(0) — replace zeros with tiny positive value
    dists = np.maximum(dists, 1e-10)

    # MLE: d_hat(x) = -k / sum log(d_j / d_k)
    log_ratios = np.log(dists[:, :-1] / dists[:, -1:])  # (n, k-1)
    lid_per_point = -(k - 1) / np.sum(log_ratios, axis=1)

    # Filter out pathological estimates (negative or very large)
    lid_per_point = lid_per_point[(lid_per_point > 0) & (lid_per_point < d)]

    return float(np.mean(lid_per_point)) if len(lid_per_point) > 0 else float("nan")


def check_lid_recovery(
    estimated_lid: float,
    true_lid: int,
    tolerance: float = 1.0,
) -> bool:
    """
    Check whether estimated LID matches true LID within tolerance.

    Used to define n*_dim: the first n at which this returns True.

    Args:
        estimated_lid: FLIPD/MLE estimate
        true_lid: known ground truth (e.g., 2 for a torus)
        tolerance: acceptable absolute error

    Returns:
        True if |estimated - true| < tolerance
    """
    return abs(estimated_lid - true_lid) < tolerance
