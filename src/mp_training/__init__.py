"""
Canonical training entrypoints for first-party studies.

This package intentionally stays small. It exposes the reusable training loop
that active studies call after they have chosen a config and a study
directory.
"""

from mp_training.checkpoint_schedule import (
    CHECKPOINT_SCHEDULE_CHOICES,
    bonnaire_1m_checkpoint_steps,
    checkpoint_schedule_metadata,
    parse_checkpoint_steps,
    resolve_checkpoint_steps,
)
from mp_training.trainer import train_run

__all__ = [
    "CHECKPOINT_SCHEDULE_CHOICES",
    "bonnaire_1m_checkpoint_steps",
    "checkpoint_schedule_metadata",
    "parse_checkpoint_steps",
    "resolve_checkpoint_steps",
    "train_run",
]
