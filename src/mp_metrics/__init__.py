"""
Canonical memorization and checkpoint-evaluation metrics.

This package exposes the two metric surfaces used in the repo:
direct tensor-vs-tensor memorization helpers for notebooks and scripts, and
checkpoint replay for full saved runs.
"""

from mp_metrics.checkpoints import evaluate_run
from mp_metrics.memorization import (
    compute_f_mem,
    compute_folder_metrics,
    compute_memorization_metrics,
    compute_nn_stats,
)

__all__ = [
    "compute_f_mem",
    "compute_folder_metrics",
    "compute_memorization_metrics",
    "compute_nn_stats",
    "evaluate_run",
]
