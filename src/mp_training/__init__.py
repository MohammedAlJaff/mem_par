"""
Canonical training entrypoints for first-party studies.

This package intentionally stays small. It exposes the reusable training loop
that active studies call after they have chosen a config and a study
directory.
"""

from mp_training.trainer import train_run

__all__ = ["train_run"]
