"""
Public metric computation helpers.

Metric modules compute metric values from in-memory tensors or features. Run
evaluation, file loading, figures, and tracking payloads live outside
`mp_metrics`.
"""

from mp_metrics.images import (
    DEFAULT_FID_FEATURE_EXTRACTOR,
    RHO_SWEEP,
    collect_image32_metric_record,
    compute_f_mem_by_rho,
    compute_image_fid,
    compute_image_memorization_metrics,
)
from mp_metrics.loss_eval import (
    DEFAULT_LOSS_EVAL_TIMES,
    collect_loss_eval_record,
    loss_eval_times,
    time_key,
)
from mp_metrics.synthetic import (
    collect_controlled_synthetic_metric_record,
    compute_f_mem,
    compute_memorization_metrics,
    compute_nn_stats,
    gmm_intrinsic_log_prob,
    gmm_intrinsic_metrics,
    intrinsic_geometry_parts,
    intrinsic_sliced_w2,
    memorization_metrics_in_spaces,
    project_to_subspace,
    synthetic_distribution_metrics,
    torus3d_metrics,
)

__all__ = [
    "DEFAULT_FID_FEATURE_EXTRACTOR",
    "DEFAULT_LOSS_EVAL_TIMES",
    "RHO_SWEEP",
    "collect_controlled_synthetic_metric_record",
    "collect_image32_metric_record",
    "collect_loss_eval_record",
    "compute_f_mem",
    "compute_f_mem_by_rho",
    "compute_image_fid",
    "compute_image_memorization_metrics",
    "compute_memorization_metrics",
    "compute_nn_stats",
    "gmm_intrinsic_log_prob",
    "gmm_intrinsic_metrics",
    "intrinsic_geometry_parts",
    "intrinsic_sliced_w2",
    "loss_eval_times",
    "memorization_metrics_in_spaces",
    "project_to_subspace",
    "synthetic_distribution_metrics",
    "time_key",
    "torus3d_metrics",
]
