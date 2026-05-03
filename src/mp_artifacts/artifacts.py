"""
Filesystem helpers for experiment/run/seed/attempt artifacts.

This module owns the on-disk layout used by saved first-party experiments.
Training and evaluation code should not invent paths directly when a helper in
this module already defines the contract.

Canonical layout:
    <experiment_dir>/
        experiment_manifest.json
        runs/
            <run_id>/
                run_manifest.json
                seeds/
                    seed_0007/
                        seed_manifest.json
                        attempts/
                            <attempt_id>/
                                status.json
                                artifacts/
                                checkpoints/
                                metrics/
                                figures/
                                logs/
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from ml_collections import ConfigDict

from mp_config import config_to_dict

TERMINAL_ATTEMPT_STATES = {"completed", "failed", "aborted"}
DONE_ATTEMPT_PREFIX = "DONE_"


def _slug(text):
    """
    Convert free-form text into a filesystem-safe identifier fragment.
    """
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text)).strip("-")


def _iso_now(now=None):
    """
    Format a timestamp as a sortable human-readable string.
    """
    if now is None:
        now = datetime.now()
    return now.isoformat(timespec="seconds")


def _relative_path(path, root):
    """
    Represent one path relative to an experiment root when possible.
    """
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _run_signature_payload(config):
    """
    Extract the seed-free semantic run signature from a resolved config.

    The goal is for one run identifier to be stable across reruns of the same
    hyperparameter configuration while ignoring seed, timestamp, and
    environment-local paths or UI choices.

    Seeds live below the run directory. This lets several random seeds share
    one semantic run identifier.
    """
    payload = config_to_dict(config)

    payload.get("training", {}).pop("seed", None)
    payload.get("training", {}).pop("show_progress", None)
    payload.get("training", {}).pop("resolved_ckpt_steps", None)
    payload.get("training", {}).pop("resume_from_attempt", None)
    payload.get("training", {}).pop("resume_from_checkpoint", None)
    payload.get("training", {}).pop("stop_after_step", None)
    payload.get("outputs", {}).pop("run_name", None)
    payload.get("study", {}).pop("name", None)
    payload.get("study", {}).pop("description", None)
    payload.get("data", {}).pop("root", None)
    payload.get("data", {}).pop("download", None)
    payload.pop("tracking", None)

    return payload


def _config_hash(config):
    """
    Hash the seed-free semantic run signature into a short stable ID.
    """
    payload = _run_signature_payload(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def make_config_hash(config):
    """
    Return the stable seed-free semantic hash used in run identifiers.

    This public wrapper lets resume validation compare a candidate config with
    the source attempt without duplicating the hash rules outside this module.
    """
    return _config_hash(config)


def _attempt_base_timestamp(now=None):
    """
    Format the base timestamp used for one attempt identifier.
    """
    if now is None:
        now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def _attempt_paths(attempt_dir):
    """
    Return the canonical child paths for one attempt directory.

    These names form the saved-run interface used by `train_run`, runner replay
    evaluation, status readers, and tests.
    """
    attempt_dir = Path(attempt_dir)
    return {
        "artifacts_dir": attempt_dir / "artifacts",
        "checkpoints_dir": attempt_dir / "checkpoints",
        "metrics_dir": attempt_dir / "metrics",
        "figures_dir": attempt_dir / "figures",
        "logs_dir": attempt_dir / "logs",
        "status_path": attempt_dir / "status.json",
        "events_log_path": attempt_dir / "logs" / "events.jsonl",
        "attempt_log_path": attempt_dir / "logs" / "attempt.log",
        "config_path": attempt_dir / "artifacts" / "config.json",
        "run_summary_path": attempt_dir / "artifacts" / "run_summary.json",
        "data_meta_path": attempt_dir / "artifacts" / "data_meta.json",
        "train_history_path": attempt_dir / "metrics" / "train_history.json",
        "checkpoint_metrics_path": attempt_dir / "metrics" / "checkpoint_metrics.json",
        "loss_eval_path": attempt_dir / "metrics" / "loss_eval.json",
        "checkpoint_sample_previews_path": (
            attempt_dir / "metrics" / "checkpoint_sample_previews.json"
        ),
    }


def _attempt_status_path_fields(attempt_dir):
    """
    Build the absolute path fields stored in an attempt `status.json`.
    """
    attempt_dir = Path(attempt_dir)
    paths = _attempt_paths(attempt_dir)
    return {
        "attempt_dir": str(attempt_dir.resolve()),
        "artifacts_dir": str(paths["artifacts_dir"].resolve()),
        "checkpoints_dir": str(paths["checkpoints_dir"].resolve()),
        "metrics_dir": str(paths["metrics_dir"].resolve()),
        "figures_dir": str(paths["figures_dir"].resolve()),
        "logs_dir": str(paths["logs_dir"].resolve()),
        "status_path": str(paths["status_path"].resolve()),
        "events_log_path": str(paths["events_log_path"].resolve()),
        "attempt_log_path": str(paths["attempt_log_path"].resolve()),
        "config_path": str(paths["config_path"].resolve()),
        "run_summary_path": str(paths["run_summary_path"].resolve()),
        "data_meta_path": str(paths["data_meta_path"].resolve()),
        "train_history_path": str(paths["train_history_path"].resolve()),
        "checkpoint_metrics_path": str(paths["checkpoint_metrics_path"].resolve()),
        "loss_eval_path": str(paths["loss_eval_path"].resolve()),
        "checkpoint_sample_previews_path": str(
            paths["checkpoint_sample_previews_path"].resolve()
        ),
    }


def _rewrite_path_strings(value, old_root, new_root):
    """
    Rewrite absolute path strings under one attempt root after a folder rename.
    """
    old_root = str(Path(old_root).resolve())
    new_root = str(Path(new_root).resolve())
    if isinstance(value, dict):
        return {
            key: _rewrite_path_strings(item, old_root, new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_path_strings(item, old_root, new_root) for item in value]
    if isinstance(value, str) and value.startswith(old_root):
        return new_root + value[len(old_root) :]
    return value


def save_json(path, payload):
    """
    Write a JSON payload to disk, creating parent directories as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _json_file_lock(path):
        _save_json_atomic(path, payload)


def load_json(path):
    """
    Load a JSON artifact from disk.
    """
    path = Path(path)
    with _json_file_lock(path):
        return _load_json_unlocked(path)


@contextlib.contextmanager
def _json_file_lock(path):
    """
    Serialize JSON readers/writers for one artifact path.

    Slurm arrays may run many attempts under the same experiment directory.
    Without an advisory lock, concurrent read/modify/write manifest updates can
    leave partial JSON on shared filesystems.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _load_json_unlocked(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _update_json_locked(path, default_factory, update_fn):
    """
    Load, mutate, and atomically save one JSON file while holding its lock.
    """
    path = Path(path)
    with _json_file_lock(path):
        payload = (
            _load_json_unlocked(path)
            if path.exists()
            else default_factory()
        )
        update_fn(payload)
        _save_json_atomic(path, payload)
        return payload


def _update_existing_json_locked(path, update_fn):
    path = Path(path)
    with _json_file_lock(path):
        if not path.exists():
            return None
        payload = _load_json_unlocked(path)
        update_fn(payload)
        _save_json_atomic(path, payload)
        return payload


def save_tensor(path, tensor):
    """
    Save a tensor artifact with `torch.save`, creating parent directories first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def make_run_id(config):
    """
    Build the stable seed-free identifier for one hyperparameter run.
    """
    base = _slug(getattr(config.outputs, "run_name", "run")) or "run"
    return f"{base}-{_config_hash(config)}"


def make_seed_id(seed):
    """
    Convert an integer seed into the canonical seed-folder name.
    """
    return f"seed_{int(seed):04d}"


def get_run_dir(experiment_dir, run_id):
    """
    Resolve the directory for one run inside an experiment root.
    """
    return Path(experiment_dir) / "runs" / str(run_id)


def get_seed_dir(experiment_dir, run_id, seed):
    """
    Resolve the directory for one seed inside a run.
    """
    return get_run_dir(experiment_dir, run_id) / "seeds" / make_seed_id(seed)


def _allocate_attempt_id(seed_dir):
    """
    Allocate a timestamp-based attempt ID without overwriting earlier attempts.
    """
    seed_dir = Path(seed_dir)
    attempts_dir = seed_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    base = _attempt_base_timestamp()
    candidate = base
    suffix = 1
    while (
        (attempts_dir / candidate).exists()
        or (attempts_dir / f"{DONE_ATTEMPT_PREFIX}{candidate}").exists()
    ):
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _default_experiment_manifest(experiment_dir, config, now):
    """
    Build the default experiment manifest payload.
    """
    experiment_dir = Path(experiment_dir)
    return {
        "experiment_id": _slug(config.study.name) or "experiment",
        "experiment_name": str(config.study.name),
        "root_dir": str(experiment_dir.resolve()),
        "created_at": now,
        "last_updated_at": now,
        "runs": {},
    }


def _default_run_manifest(experiment_dir, run_dir, run_id, config, now):
    """
    Build the default run manifest payload.
    """
    experiment_dir = Path(experiment_dir)
    run_dir = Path(run_dir)
    return {
        "run_id": run_id,
        "run_name": str(config.outputs.run_name),
        "experiment_name": str(config.study.name),
        "config_hash": _config_hash(config),
        "resolved_config_without_seed": _run_signature_payload(config),
        "path": _relative_path(run_dir, experiment_dir),
        "manifest_path": _relative_path(run_dir / "run_manifest.json", experiment_dir),
        "created_at": now,
        "last_updated_at": now,
        "seeds": {},
    }


def _default_seed_manifest(experiment_dir, seed_dir, run_id, seed, now):
    """
    Build the default seed manifest payload.
    """
    experiment_dir = Path(experiment_dir)
    seed_dir = Path(seed_dir)
    return {
        "run_id": run_id,
        "seed": int(seed),
        "seed_id": seed_dir.name,
        "path": _relative_path(seed_dir, experiment_dir),
        "manifest_path": _relative_path(seed_dir / "seed_manifest.json", experiment_dir),
        "created_at": now,
        "last_updated_at": now,
        "latest_attempt": None,
        "latest_completed_attempt": None,
        "attempts": {},
    }


def _layout_context(attempt_dir):
    """
    Recover experiment, run, and seed directories from one attempt path.
    """
    attempt_dir = Path(attempt_dir).resolve()
    attempts_dir = attempt_dir.parent
    seed_dir = attempts_dir.parent
    run_dir = seed_dir.parent.parent
    experiment_dir = run_dir.parent.parent
    return {
        "experiment_dir": experiment_dir,
        "run_dir": run_dir,
        "seed_dir": seed_dir,
        "attempt_dir": attempt_dir,
        "run_id": run_dir.name,
        "seed_id": seed_dir.name,
        "attempt_id": attempt_dir.name,
    }


def _append_event_log(attempt_dir, payload):
    """
    Append one lifecycle event record under `logs/events.jsonl`.
    """
    paths = _attempt_paths(attempt_dir)
    events_path = paths["events_log_path"]
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def load_attempt_status(attempt_dir):
    """
    Load the live `status.json` payload for one attempt directory.
    """
    return load_json(_attempt_paths(attempt_dir)["status_path"])


def patch_attempt_status(attempt_dir, **fields):
    """
    Update attempt status fields without implying a lifecycle state transition.
    """
    attempt_dir = Path(attempt_dir)
    status_path = _attempt_paths(attempt_dir)["status_path"]
    status = load_json(status_path)
    now = _iso_now()

    status.update(fields)
    status["updated_at"] = now
    if status.get("state") in TERMINAL_ATTEMPT_STATES and status.get("finished_at") is None:
        status["finished_at"] = now

    save_json(status_path, status)
    _sync_status_into_manifests(attempt_dir, status)
    return status


def append_attempt_event(attempt_dir, event_type, **fields):
    """
    Append one structured event record under `logs/events.jsonl`.
    """
    payload = {
        "timestamp": _iso_now(),
        "event_type": str(event_type),
    }
    payload.update(fields)
    _append_event_log(attempt_dir, payload)
    return payload


def append_attempt_log(attempt_dir, line, *, timestamp=None):
    """
    Append one human-readable line to `logs/attempt.log`.
    """
    paths = _attempt_paths(attempt_dir)
    log_path = paths["attempt_log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = timestamp or _iso_now()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{prefix}] {line}\n")
    return log_path


def _register_attempt_in_manifests(experiment_dir, config, run_id, seed, attempt_dir, now):
    """
    Create or update experiment/run/seed manifests for one new attempt.
    """
    experiment_dir = Path(experiment_dir)
    attempt_dir = Path(attempt_dir)
    context = _layout_context(attempt_dir)
    run_dir = context["run_dir"]
    seed_dir = context["seed_dir"]
    attempt_id = context["attempt_id"]
    seed_id = context["seed_id"]

    experiment_manifest_path = experiment_dir / "experiment_manifest.json"

    def update_experiment_manifest(experiment_manifest):
        experiment_manifest["experiment_name"] = str(config.study.name)
        experiment_manifest["experiment_id"] = _slug(config.study.name) or "experiment"
        experiment_manifest["root_dir"] = str(experiment_dir.resolve())
        experiment_manifest.setdefault("created_at", now)
        experiment_manifest["last_updated_at"] = now
        experiment_manifest.setdefault("runs", {})
        experiment_manifest["runs"][run_id] = {
            "path": _relative_path(run_dir, experiment_dir),
            "manifest_path": _relative_path(run_dir / "run_manifest.json", experiment_dir),
            "run_name": str(config.outputs.run_name),
            "config_hash": _config_hash(config),
            "last_updated_at": now,
        }

    _update_json_locked(
        experiment_manifest_path,
        lambda: _default_experiment_manifest(experiment_dir, config, now),
        update_experiment_manifest,
    )

    run_manifest_path = run_dir / "run_manifest.json"

    def update_run_manifest(run_manifest):
        run_manifest["last_updated_at"] = now
        run_manifest.setdefault("seeds", {})
        run_manifest["seeds"].setdefault(
            seed_id,
            {
                "seed": int(seed),
                "path": _relative_path(seed_dir, experiment_dir),
                "manifest_path": _relative_path(seed_dir / "seed_manifest.json", experiment_dir),
                "latest_attempt": None,
                "latest_completed_attempt": None,
                "last_updated_at": now,
            },
        )
        run_manifest["seeds"][seed_id]["last_updated_at"] = now

    _update_json_locked(
        run_manifest_path,
        lambda: _default_run_manifest(experiment_dir, run_dir, run_id, config, now),
        update_run_manifest,
    )

    seed_manifest_path = seed_dir / "seed_manifest.json"

    def update_seed_manifest(seed_manifest):
        seed_manifest["last_updated_at"] = now
        seed_manifest["latest_attempt"] = attempt_id
        seed_manifest.setdefault("attempts", {})
        seed_manifest["attempts"][attempt_id] = {
            "path": _relative_path(attempt_dir, experiment_dir),
            "status_path": _relative_path(attempt_dir / "status.json", experiment_dir),
            "state": "created",
            "created_at": now,
            "updated_at": now,
        }

    _update_json_locked(
        seed_manifest_path,
        lambda: _default_seed_manifest(experiment_dir, seed_dir, run_id, seed, now),
        update_seed_manifest,
    )


def _sync_status_into_manifests(attempt_dir, status):
    """
    Propagate one attempt status update into the seed/run/experiment manifests.
    """
    context = _layout_context(attempt_dir)
    experiment_dir = context["experiment_dir"]
    run_dir = context["run_dir"]
    seed_dir = context["seed_dir"]
    attempt_id = context["attempt_id"]
    run_id = context["run_id"]
    seed_id = context["seed_id"]

    seed_manifest_path = seed_dir / "seed_manifest.json"
    def update_seed_manifest(seed_manifest):
        seed_manifest["last_updated_at"] = status["updated_at"]
        seed_manifest["latest_attempt"] = attempt_id
        seed_manifest.setdefault("attempts", {})
        seed_manifest["attempts"].setdefault(
            attempt_id,
            {
                "path": _relative_path(attempt_dir, experiment_dir),
                "status_path": _relative_path(attempt_dir / "status.json", experiment_dir),
                "created_at": status.get("created_at", status["updated_at"]),
            },
        )
        attempt_entry = seed_manifest["attempts"][attempt_id]
        attempt_entry["state"] = status["state"]
        attempt_entry["updated_at"] = status["updated_at"]
        if status.get("finished_at") is not None:
            attempt_entry["finished_at"] = status["finished_at"]
        if status.get("last_error") is not None:
            attempt_entry["last_error"] = status["last_error"]
        if status["state"] == "completed":
            seed_manifest["latest_completed_attempt"] = attempt_id

    _update_existing_json_locked(seed_manifest_path, update_seed_manifest)

    run_manifest_path = run_dir / "run_manifest.json"
    def update_run_manifest(run_manifest):
        run_manifest["last_updated_at"] = status["updated_at"]
        run_manifest.setdefault("seeds", {})
        run_manifest["seeds"].setdefault(
            seed_id,
            {
                "seed": int(status["seed"]),
                "path": _relative_path(seed_dir, experiment_dir),
                "manifest_path": _relative_path(seed_dir / "seed_manifest.json", experiment_dir),
            },
        )
        seed_entry = run_manifest["seeds"][seed_id]
        seed_entry["latest_attempt"] = attempt_id
        seed_entry["last_updated_at"] = status["updated_at"]
        if status["state"] == "completed":
            seed_entry["latest_completed_attempt"] = attempt_id

    _update_existing_json_locked(run_manifest_path, update_run_manifest)

    experiment_manifest_path = experiment_dir / "experiment_manifest.json"
    def update_experiment_manifest(experiment_manifest):
        experiment_manifest["last_updated_at"] = status["updated_at"]
        experiment_manifest.setdefault("runs", {})
        experiment_manifest["runs"].setdefault(
            run_id,
            {
                "path": _relative_path(run_dir, experiment_dir),
                "manifest_path": _relative_path(run_dir / "run_manifest.json", experiment_dir),
                "run_name": status["run_name"],
                "config_hash": status["config_hash"],
            },
        )
        experiment_manifest["runs"][run_id]["last_updated_at"] = status["updated_at"]

    _update_existing_json_locked(experiment_manifest_path, update_experiment_manifest)


def _remove_old_attempt_manifest_entry(attempt_dir, old_attempt_id, new_attempt_id):
    """
    Remove stale manifest entries after an attempt folder receives DONE_.
    """
    if not old_attempt_id or old_attempt_id == new_attempt_id:
        return

    context = _layout_context(attempt_dir)
    experiment_dir = context["experiment_dir"]
    run_dir = context["run_dir"]
    seed_dir = context["seed_dir"]
    seed_id = context["seed_id"]

    seed_manifest_path = seed_dir / "seed_manifest.json"
    def update_seed_manifest(seed_manifest):
        seed_manifest.get("attempts", {}).pop(old_attempt_id, None)
        if seed_manifest.get("latest_attempt") == old_attempt_id:
            seed_manifest["latest_attempt"] = new_attempt_id
        if seed_manifest.get("latest_completed_attempt") == old_attempt_id:
            seed_manifest["latest_completed_attempt"] = new_attempt_id

    _update_existing_json_locked(seed_manifest_path, update_seed_manifest)

    run_manifest_path = run_dir / "run_manifest.json"
    def update_run_manifest(run_manifest):
        seed_entry = run_manifest.get("seeds", {}).get(seed_id)
        if seed_entry is not None:
            if seed_entry.get("latest_attempt") == old_attempt_id:
                seed_entry["latest_attempt"] = new_attempt_id
            if seed_entry.get("latest_completed_attempt") == old_attempt_id:
                seed_entry["latest_completed_attempt"] = new_attempt_id

    _update_existing_json_locked(run_manifest_path, update_run_manifest)

    experiment_manifest_path = experiment_dir / "experiment_manifest.json"
    def update_experiment_manifest(experiment_manifest):
        experiment_manifest["last_updated_at"] = _iso_now()

    _update_existing_json_locked(experiment_manifest_path, update_experiment_manifest)


def _refresh_run_summary_paths(attempt_dir, old_attempt_dir, original_attempt_id):
    """
    Rewrite run-summary paths after a successful-training DONE_ rename.
    """
    attempt_dir = Path(attempt_dir)
    summary_path = _attempt_paths(attempt_dir)["run_summary_path"]
    if not summary_path.exists():
        return None

    summary = load_json(summary_path)
    summary = _rewrite_path_strings(summary, old_attempt_dir, attempt_dir)
    summary.update(
        {
            "run_dir": str(attempt_dir.resolve()),
            "attempt_id": attempt_dir.name,
            "original_attempt_id": original_attempt_id,
            "artifacts_dir": str((attempt_dir / "artifacts").resolve()),
            "checkpoints_dir": str((attempt_dir / "checkpoints").resolve()),
            "metrics_dir": str((attempt_dir / "metrics").resolve()),
            "figures_dir": str((attempt_dir / "figures").resolve()),
            "logs_dir": str((attempt_dir / "logs").resolve()),
            "status_path": str((attempt_dir / "status.json").resolve()),
            "attempt_log_path": str((attempt_dir / "logs" / "attempt.log").resolve()),
        }
    )
    save_json(summary_path, summary)
    return summary


def mark_attempt_done(attempt_dir):
    """
    Rename a successfully trained attempt folder with a DONE_ prefix.

    The marker is applied only when callers explicitly invoke this helper on
    the successful path. Failed and aborted attempts remain at their original
    paths.
    """
    attempt_dir = Path(attempt_dir).resolve()
    if attempt_dir.name.startswith(DONE_ATTEMPT_PREFIX):
        status = load_attempt_status(attempt_dir)
        status["attempt_id"] = attempt_dir.name
        status.update(_attempt_status_path_fields(attempt_dir))
        save_json(_attempt_paths(attempt_dir)["status_path"], status)
        _sync_status_into_manifests(attempt_dir, status)
        return attempt_dir

    done_attempt_dir = attempt_dir.with_name(f"{DONE_ATTEMPT_PREFIX}{attempt_dir.name}")
    if done_attempt_dir.exists():
        raise FileExistsError(f"DONE attempt directory already exists: {done_attempt_dir}")

    status = load_attempt_status(attempt_dir)
    original_attempt_id = (
        status.get("original_attempt_id")
        or status.get("attempt_id")
        or attempt_dir.name
    )

    old_attempt_dir = attempt_dir
    attempt_dir.rename(done_attempt_dir)

    status = _rewrite_path_strings(status, old_attempt_dir, done_attempt_dir)
    status["attempt_id"] = done_attempt_dir.name
    status["original_attempt_id"] = original_attempt_id
    status["updated_at"] = _iso_now()
    status.update(_attempt_status_path_fields(done_attempt_dir))
    save_json(_attempt_paths(done_attempt_dir)["status_path"], status)

    _refresh_run_summary_paths(
        done_attempt_dir,
        old_attempt_dir,
        original_attempt_id,
    )
    _remove_old_attempt_manifest_entry(
        done_attempt_dir,
        original_attempt_id,
        done_attempt_dir.name,
    )
    _sync_status_into_manifests(done_attempt_dir, status)
    _append_event_log(
        done_attempt_dir,
        {
            "timestamp": _iso_now(),
            "event_type": "attempt_folder_marked_done",
            "state": status.get("state"),
            "attempt_id": done_attempt_dir.name,
            "original_attempt_id": original_attempt_id,
        },
    )
    return done_attempt_dir


def create_run_layout(experiment_dir, config):
    """
    Create the canonical attempt directory tree for one training launch.

    The returned path is the attempt directory itself, not a flat run folder.
    All later training, evaluation, and status files are written below this
    directory.
    """
    experiment_dir = Path(experiment_dir)
    run_id = make_run_id(config)
    seed = int(config.training.seed)
    seed_id = make_seed_id(seed)
    run_dir = get_run_dir(experiment_dir, run_id)
    seed_dir = run_dir / "seeds" / seed_id
    attempt_id = _allocate_attempt_id(seed_dir)
    attempt_dir = seed_dir / "attempts" / attempt_id
    paths = _attempt_paths(attempt_dir)

    for path in (
        experiment_dir,
        run_dir,
        seed_dir,
        paths["artifacts_dir"],
        paths["checkpoints_dir"],
        paths["metrics_dir"],
        paths["figures_dir"],
        paths["logs_dir"],
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

    now = _iso_now()
    _register_attempt_in_manifests(experiment_dir, config, run_id, seed, attempt_dir, now)

    status = {
        "state": "created",
        "current_stage": "created",
        "experiment_id": _slug(config.study.name) or "experiment",
        "experiment_name": str(config.study.name),
        "run_id": run_id,
        "run_name": str(config.outputs.run_name),
        "config_hash": _config_hash(config),
        "seed": seed,
        "seed_id": seed_id,
        "optimizer": str(config.training.optimizer),
        "lr": float(config.training.lr),
        "weight_decay": float(config.training.weight_decay),
        "momentum": float(config.training.momentum),
        "attempt_id": attempt_id,
        "created_at": now,
        "updated_at": now,
        "finished_at": None,
        "attempt_dir": str(attempt_dir.resolve()),
        "artifacts_dir": str(paths["artifacts_dir"].resolve()),
        "checkpoints_dir": str(paths["checkpoints_dir"].resolve()),
        "metrics_dir": str(paths["metrics_dir"].resolve()),
        "figures_dir": str(paths["figures_dir"].resolve()),
        "logs_dir": str(paths["logs_dir"].resolve()),
        "status_path": str(paths["status_path"].resolve()),
        "events_log_path": str(paths["events_log_path"].resolve()),
        "attempt_log_path": str(paths["attempt_log_path"].resolve()),
        "config_path": str(paths["config_path"].resolve()),
        "run_summary_path": str(paths["run_summary_path"].resolve()),
        "data_meta_path": str(paths["data_meta_path"].resolve()),
        "train_history_path": str(paths["train_history_path"].resolve()),
        "checkpoint_metrics_path": str(paths["checkpoint_metrics_path"].resolve()),
        "loss_eval_path": str(paths["loss_eval_path"].resolve()),
        "checkpoint_sample_previews_path": str(
            paths["checkpoint_sample_previews_path"].resolve()
        ),
        "progress": {},
        "latest_metrics": {},
        "last_checkpoint": None,
        "evaluation_progress": {},
        "generated_figures": {},
        "wandb": {
            "enabled": False,
            "active": False,
            "mode": None,
            "run_id": None,
            "run_name": None,
            "run_path": None,
            "run_url": None,
            "dir": None,
            "warning": None,
        },
        "warnings": [],
        "last_error": None,
    }
    save_json(paths["status_path"], status)
    _append_event_log(
        attempt_dir,
        {
            "timestamp": now,
            "event_type": "state_transition",
            "state": "created",
            "attempt_id": attempt_id,
        },
    )
    return attempt_dir


def update_attempt_status(attempt_dir, state, **fields):
    """
    Update lifecycle metadata for one attempt and sync its manifests.

    `status.json` is the live view for one attempt. The experiment, run, and
    seed manifests receive compact copies so a user can inspect progress
    without opening every attempt directory.
    """
    attempt_dir = Path(attempt_dir)
    status_path = _attempt_paths(attempt_dir)["status_path"]
    status = load_json(status_path)
    now = _iso_now()

    status["state"] = state
    status["current_stage"] = state
    status["updated_at"] = now
    if state in TERMINAL_ATTEMPT_STATES:
        status["finished_at"] = now
    status.update(fields)

    save_json(status_path, status)
    _append_event_log(
        attempt_dir,
        {
            "timestamp": now,
            "event_type": "state_transition",
            "state": state,
            "attempt_id": status["attempt_id"],
            "last_error": status.get("last_error"),
        },
    )
    _sync_status_into_manifests(attempt_dir, status)
    return status


def save_config_snapshot(run_dir, config):
    """
    Serialize the resolved config into the attempt's artifact directory.

    Evaluation replay loads this file rather than relying on current defaults.
    """
    payload = config_to_dict(config)
    save_json(Path(run_dir) / "artifacts" / "config.json", payload)


def load_config_snapshot(run_dir):
    """
    Rebuild a `ConfigDict` from a saved attempt-directory config snapshot.
    """
    payload = load_json(Path(run_dir) / "artifacts" / "config.json")
    return ConfigDict(payload)


def save_checkpoint(run_dir, step, model, optimizer):
    """
    Save one training checkpoint containing model and optimizer state.

    Checkpoint filenames use optimization step numbers because step is the
    shared x-axis for training loss, generated samples, and evaluation metrics.
    """
    path = Path(run_dir) / "checkpoints" / f"step_{int(step):06d}.pt"
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return path


def _checkpoint_path_step(path):
    """
    Extract the integer training step from a checkpoint filename.

    Checkpoints are normally zero-padded, but long runs can exceed the original
    width, e.g. `step_1000000.pt`. Numeric sorting keeps the final checkpoint
    last even when filename lengths differ.
    """
    path = Path(path)
    match = re.fullmatch(r"step_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint filename: {path}")
    return int(match.group(1))


def list_checkpoint_paths(run_dir):
    """
    Return the saved checkpoint files in chronological order.
    """
    return sorted(
        (Path(run_dir) / "checkpoints").glob("step_*.pt"),
        key=_checkpoint_path_step,
    )


def find_latest_attempt_dir(experiment_dir, run_id, seed):
    """
    Resolve the newest attempt directory for one specific run and seed.
    """
    seed_dir = get_seed_dir(experiment_dir, run_id, seed)
    seed_manifest_path = seed_dir / "seed_manifest.json"
    if not seed_manifest_path.exists():
        raise FileNotFoundError(f"No seed manifest found at {seed_manifest_path}")

    seed_manifest = load_json(seed_manifest_path)
    latest_attempt = seed_manifest.get("latest_attempt")
    if not latest_attempt:
        raise FileNotFoundError(f"No attempts recorded for {seed_dir}")
    return seed_dir / "attempts" / latest_attempt


def find_latest_completed_attempt_dir(experiment_dir, run_id, seed):
    """
    Resolve the newest completed attempt directory for one specific run and seed.
    """
    seed_dir = get_seed_dir(experiment_dir, run_id, seed)
    seed_manifest_path = seed_dir / "seed_manifest.json"
    if not seed_manifest_path.exists():
        raise FileNotFoundError(f"No seed manifest found at {seed_manifest_path}")

    seed_manifest = load_json(seed_manifest_path)
    latest_attempt = seed_manifest.get("latest_completed_attempt")
    if not latest_attempt:
        raise FileNotFoundError(f"No completed attempts recorded for {seed_dir}")
    return seed_dir / "attempts" / latest_attempt


def _find_latest_attempt_across_experiment(experiment_dir, *, completed_only):
    """
    Find the newest attempt under an experiment root by reading status files.
    """
    experiment_dir = Path(experiment_dir)
    newest = None

    for status_path in experiment_dir.glob("runs/*/seeds/*/attempts/*/status.json"):
        status = load_json(status_path)
        if completed_only and status.get("state") != "completed":
            continue

        attempt_dir = status_path.parent
        marker = status.get("finished_at") or status.get("updated_at") or status.get(
            "created_at"
        )
        candidate = (marker, attempt_dir)
        if newest is None or candidate > newest:
            newest = candidate

    if newest is None:
        raise FileNotFoundError(
            f"No {'completed ' if completed_only else ''}attempts found under {experiment_dir}"
        )
    return newest[1]


def find_latest_run_dir(experiment_dir):
    """
    Compatibility helper returning the newest completed attempt in an experiment.
    """
    return _find_latest_attempt_across_experiment(experiment_dir, completed_only=True)


def resolve_attempt_dir(
    experiment_dir,
    *,
    run_id=None,
    seed=None,
    attempt_dir=None,
    completed=False,
):
    """
    Resolve one attempt directory explicitly or from run/seed identity.
    """
    if attempt_dir is not None:
        return Path(attempt_dir)
    if run_id is None or seed is None:
        raise ValueError("resolve_attempt_dir requires run_id and seed when attempt_dir is absent")
    if completed:
        return find_latest_completed_attempt_dir(experiment_dir, run_id, seed)
    return find_latest_attempt_dir(experiment_dir, run_id, seed)


def resolve_run_dir(experiment_dir, run_dir=None):
    """
    Compatibility helper returning an explicit attempt dir or the latest completed one.
    """
    if run_dir is not None:
        return Path(run_dir)
    return find_latest_run_dir(experiment_dir)


if __name__ == "__main__":
    import tempfile

    from mp_config import get_base_config

    print("mp_artifacts.artifacts example")

    config = get_base_config()
    config.study.name = "artifact_demo"
    config.outputs.run_name = "artifact_demo"
    config.training.seed = 7

    with tempfile.TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / "artifact_demo"
        attempt_dir = create_run_layout(experiment_dir, config)

        save_json(attempt_dir / "artifacts" / "demo.json", {"hello": "world"})
        save_tensor(attempt_dir / "artifacts" / "demo_tensor.pt", torch.arange(6))
        save_config_snapshot(attempt_dir, config)
        update_attempt_status(attempt_dir, "completed")

        latest_completed = find_latest_completed_attempt_dir(
            experiment_dir,
            make_run_id(config),
            config.training.seed,
        )
        reloaded = load_config_snapshot(attempt_dir)

        print(f"attempt_dir          : {attempt_dir}")
        print(f"latest_completed     : {latest_completed}")
        print(f"run_manifest         : {attempt_dir.parents[3] / 'run_manifest.json'}")
        print(f"seed_manifest        : {attempt_dir.parents[1] / 'seed_manifest.json'}")
        print(f"reloaded study name  : {reloaded.study.name}")
