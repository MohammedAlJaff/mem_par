"""
Filesystem helpers for experiment/run/seed/attempt artifacts.

This module owns the on-disk layout used by saved first-party experiments.

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

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from ml_collections import ConfigDict

from mp_config import config_to_dict

TERMINAL_ATTEMPT_STATES = {"completed", "failed", "aborted"}


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
    """
    payload = config_to_dict(config)

    payload.get("training", {}).pop("seed", None)
    payload.get("training", {}).pop("show_progress", None)
    payload.get("outputs", {}).pop("run_name", None)
    payload.get("study", {}).pop("name", None)
    payload.get("study", {}).pop("description", None)
    payload.get("data", {}).pop("root", None)
    payload.get("data", {}).pop("download", None)

    return payload


def _config_hash(config):
    """
    Hash the seed-free semantic run signature into a short stable ID.
    """
    payload = _run_signature_payload(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


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
        "config_path": attempt_dir / "artifacts" / "config.json",
        "run_summary_path": attempt_dir / "artifacts" / "run_summary.json",
        "data_meta_path": attempt_dir / "artifacts" / "data_meta.json",
        "train_history_path": attempt_dir / "metrics" / "train_history.json",
        "checkpoint_metrics_path": attempt_dir / "metrics" / "checkpoint_metrics.json",
    }


def save_json(path, payload):
    """
    Write a JSON payload to disk, creating parent directories as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_json(path):
    """
    Load a JSON artifact from disk.
    """
    return json.loads(Path(path).read_text())


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
    while (attempts_dir / candidate).exists():
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
    experiment_manifest = (
        load_json(experiment_manifest_path)
        if experiment_manifest_path.exists()
        else _default_experiment_manifest(experiment_dir, config, now)
    )
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
    save_json(experiment_manifest_path, experiment_manifest)

    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = (
        load_json(run_manifest_path)
        if run_manifest_path.exists()
        else _default_run_manifest(experiment_dir, run_dir, run_id, config, now)
    )
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
    save_json(run_manifest_path, run_manifest)

    seed_manifest_path = seed_dir / "seed_manifest.json"
    seed_manifest = (
        load_json(seed_manifest_path)
        if seed_manifest_path.exists()
        else _default_seed_manifest(experiment_dir, seed_dir, run_id, seed, now)
    )
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
    save_json(seed_manifest_path, seed_manifest)


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
    if seed_manifest_path.exists():
        seed_manifest = load_json(seed_manifest_path)
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
        save_json(seed_manifest_path, seed_manifest)

    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = load_json(run_manifest_path)
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
        save_json(run_manifest_path, run_manifest)

    experiment_manifest_path = experiment_dir / "experiment_manifest.json"
    if experiment_manifest_path.exists():
        experiment_manifest = load_json(experiment_manifest_path)
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
        save_json(experiment_manifest_path, experiment_manifest)


def create_run_layout(experiment_dir, config):
    """
    Create the canonical attempt directory tree for one training launch.

    The returned path is the attempt directory itself, not a flat run folder.
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
        "experiment_id": _slug(config.study.name) or "experiment",
        "experiment_name": str(config.study.name),
        "run_id": run_id,
        "run_name": str(config.outputs.run_name),
        "config_hash": _config_hash(config),
        "seed": seed,
        "seed_id": seed_id,
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
        "config_path": str(paths["config_path"].resolve()),
        "run_summary_path": str(paths["run_summary_path"].resolve()),
        "data_meta_path": str(paths["data_meta_path"].resolve()),
        "train_history_path": str(paths["train_history_path"].resolve()),
        "checkpoint_metrics_path": str(paths["checkpoint_metrics_path"].resolve()),
        "last_error": None,
    }
    save_json(paths["status_path"], status)
    _append_event_log(
        attempt_dir,
        {
            "timestamp": now,
            "state": "created",
            "attempt_id": attempt_id,
        },
    )
    return attempt_dir


def update_attempt_status(attempt_dir, state, **fields):
    """
    Update the lifecycle metadata for one attempt and sync its manifests.
    """
    attempt_dir = Path(attempt_dir)
    status_path = _attempt_paths(attempt_dir)["status_path"]
    status = load_json(status_path)
    now = _iso_now()

    status["state"] = state
    status["updated_at"] = now
    if state in TERMINAL_ATTEMPT_STATES:
        status["finished_at"] = now
    status.update(fields)

    save_json(status_path, status)
    _append_event_log(
        attempt_dir,
        {
            "timestamp": now,
            "state": state,
            "attempt_id": status["attempt_id"],
            "last_error": status.get("last_error"),
        },
    )
    _sync_status_into_manifests(attempt_dir, status)
    return status


def save_config_snapshot(run_dir, config):
    """
    Serialize the active config into the attempt's artifact directory.
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


def list_checkpoint_paths(run_dir):
    """
    Return the saved checkpoint files in chronological order.
    """
    return sorted((Path(run_dir) / "checkpoints").glob("step_*.pt"))


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
