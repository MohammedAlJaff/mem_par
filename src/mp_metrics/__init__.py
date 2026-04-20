"""
Canonical memorization and checkpoint-evaluation metrics.

This package exposes the two metric surfaces used in the repo:
direct tensor-vs-tensor memorization helpers for notebooks and scripts, and
checkpoint evaluation for full saved runs.
"""

from mp_metrics.checkpoints import evaluate_run
from mp_metrics.checkpoint_previews import (
    DEFAULT_CHECKPOINT_PREVIEW_COUNT,
    checkpoint_preview_figure_name,
    collect_checkpoint_sample_preview_record,
)
from mp_metrics.images import (
    compute_f_mem_by_threshold,
    compute_image_fid,
    compute_image_folder_fid,
    compute_image_folder_memorization_metrics,
    compute_image_memorization_metrics,
)
from mp_metrics.loss_eval import (
    DEFAULT_LOSS_EVAL_TIMES,
    collect_loss_eval_record,
    loss_eval_metric_payload,
    loss_eval_times,
    time_key,
)
from mp_metrics.memorization import (
    compute_f_mem,
    compute_folder_metrics,
    compute_memorization_metrics,
    compute_nn_stats,
)

__all__ = [
    "compute_f_mem",
    "compute_f_mem_by_threshold",
    "compute_folder_metrics",
    "compute_image_fid",
    "compute_image_folder_fid",
    "compute_image_folder_memorization_metrics",
    "compute_image_memorization_metrics",
    "compute_memorization_metrics",
    "compute_nn_stats",
    "DEFAULT_CHECKPOINT_PREVIEW_COUNT",
    "DEFAULT_LOSS_EVAL_TIMES",
    "checkpoint_preview_figure_name",
    "collect_checkpoint_sample_preview_record",
    "evaluate_run",
    "collect_loss_eval_record",
    "loss_eval_metric_payload",
    "loss_eval_times",
    "time_key",
]
