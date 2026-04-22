"""
Compatibility import path for controlled synthetic metrics.

The metric implementation now lives beside the direct synthetic runner in
`prototyping.runners.metrics_synthetic_controlled`. Older notebooks and tests
still import `prototyping.metrics_synthetic_controlled`, so this narrow shim
keeps that public path working while the prototype tree settles.
"""

from prototyping.runners.metrics_synthetic_controlled import (
    collect_controlled_synthetic_metric_record,
    gmm_intrinsic_log_prob,
    gmm_intrinsic_metrics,
    intrinsic_sliced_w2,
    memorization_metrics_in_spaces,
    project_to_subspace,
    synthetic_distribution_metrics,
    torus3d_metrics,
)


__all__ = [
    "collect_controlled_synthetic_metric_record",
    "gmm_intrinsic_log_prob",
    "gmm_intrinsic_metrics",
    "intrinsic_sliced_w2",
    "memorization_metrics_in_spaces",
    "project_to_subspace",
    "synthetic_distribution_metrics",
    "torus3d_metrics",
]
