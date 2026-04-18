"""
Compatibility import path for controlled synthetic metrics.

The metric implementation now lives beside the direct synthetic runner in
`prototyping.runners.metrics_synthetic_controlled`. Older notebooks and tests
still import `prototyping.metrics_synthetic_controlled`, so this narrow shim
keeps that public path working while the prototype tree settles.
"""

from prototyping.runners.metrics_synthetic_controlled import (
    collect_metric_record,
    component_coverage_metrics,
    gmm_latent_log_prob,
    latent_frechet_distance,
    latent_sliced_w2,
    memorization_metrics_in_spaces,
    oracle_gmm_metrics,
    project_to_subspace,
)


__all__ = [
    "collect_metric_record",
    "component_coverage_metrics",
    "gmm_latent_log_prob",
    "latent_frechet_distance",
    "latent_sliced_w2",
    "memorization_metrics_in_spaces",
    "oracle_gmm_metrics",
    "project_to_subspace",
]
