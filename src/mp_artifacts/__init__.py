"""
Artifact helpers for study runs.

This package owns the small but important pieces of run persistence:
directory layout, config snapshots, JSON summaries, tensors, and checkpoints.
It does not decide *what* to train or *how* to evaluate a model; it only
provides the filesystem contract that lets other packages save and reload
their work consistently.
"""

from mp_artifacts.artifacts import (
    append_attempt_event,
    append_attempt_log,
    create_run_layout,
    find_latest_attempt_dir,
    find_latest_completed_attempt_dir,
    find_latest_run_dir,
    get_run_dir,
    get_seed_dir,
    list_checkpoint_paths,
    load_attempt_status,
    load_config_snapshot,
    load_json,
    make_run_id,
    make_seed_id,
    patch_attempt_status,
    resolve_attempt_dir,
    resolve_run_dir,
    save_checkpoint,
    save_config_snapshot,
    save_json,
    save_tensor,
    update_attempt_status,
)

__all__ = [
    "append_attempt_event",
    "append_attempt_log",
    "create_run_layout",
    "find_latest_attempt_dir",
    "find_latest_completed_attempt_dir",
    "find_latest_run_dir",
    "get_run_dir",
    "get_seed_dir",
    "list_checkpoint_paths",
    "load_attempt_status",
    "load_config_snapshot",
    "load_json",
    "make_run_id",
    "make_seed_id",
    "patch_attempt_status",
    "resolve_attempt_dir",
    "resolve_run_dir",
    "save_checkpoint",
    "save_config_snapshot",
    "save_json",
    "save_tensor",
    "update_attempt_status",
]
