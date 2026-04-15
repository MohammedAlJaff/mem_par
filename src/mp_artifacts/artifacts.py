"""
Filesystem helpers for study runs and saved artifacts.

This file defines the on-disk layout used by saved runs.

Canonical run layout:
    runs/<run_id>/
        checkpoints/   model + optimizer snapshots over time
        artifacts/     config, tensors, indices, and replay-critical metadata
        metrics/       summaries produced during or after training
"""

import json
import re
from datetime import datetime
from pathlib import Path

import torch
from ml_collections import ConfigDict

from mp_config import config_to_dict


def _slug(text):
    """
    Convert free-form text into a filesystem-safe run-name fragment.
    """
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text)).strip("-")


def make_run_id(config):
    """
    Build the canonical run identifier for one study launch.

    We include the run name, prediction/loss choices, seed, and a timestamp so
    a researcher can tell what the run was before opening any files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _slug(config.outputs.run_name)
    return (
        f"{base}_pred-{_slug(config.framework.prediction_space)}"
        f"_loss-{_slug(config.framework.loss_space)}"
        f"_seed-{int(config.training.seed)}_{timestamp}"
    )


def create_run_layout(study_dir, config):
    """
    Create the canonical directory tree for one study run.

    Layout:
        runs/<run_id>/
            checkpoints/
            artifacts/
            metrics/

    We also maintain `runs/LATEST` so simple scripts and notebooks can resolve
    the most recent run without searching by hand.
    """
    study_dir = Path(study_dir)
    runs_dir = study_dir / "runs"
    figures_dir = study_dir / "figures"
    runs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_id = make_run_id(config)
    run_dir = runs_dir / run_id
    checkpoints_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"
    metrics_dir = run_dir / "metrics"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (runs_dir / "LATEST").write_text(run_id)
    return run_dir


def find_latest_run_dir(study_dir):
    """
    Resolve the run directory pointed to by the `runs/LATEST` marker.
    """
    latest_file = Path(study_dir) / "runs" / "LATEST"
    if not latest_file.exists():
        raise FileNotFoundError(f"No latest run marker found at {latest_file}")
    run_id = latest_file.read_text().strip()
    run_dir = Path(study_dir) / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Latest run directory not found: {run_dir}")
    return run_dir


def resolve_run_dir(study_dir, run_dir=None):
    """
    Use an explicit run directory if provided, otherwise fall back to `LATEST`.
    """
    if run_dir is not None:
        return Path(run_dir)
    return find_latest_run_dir(study_dir)


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


def save_config_snapshot(run_dir, config):
    """
    Serialize the active config into the run's artifact directory.

    Save the exact study settings used for this run.
    """
    payload = config_to_dict(config)
    save_json(Path(run_dir) / "artifacts" / "config.json", payload)


def load_config_snapshot(run_dir):
    """
    Rebuild a `ConfigDict` from a saved run-directory config snapshot.
    """
    payload = load_json(Path(run_dir) / "artifacts" / "config.json")
    return ConfigDict(payload)


def save_checkpoint(run_dir, step, model, optimizer):
    """
    Save one training checkpoint containing model and optimizer state.

    Checkpoints are named by step so later evaluation can replay the entire
    training trajectory in chronological order.
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


if __name__ == "__main__":
    import tempfile

    from mp_config import get_base_config

    print("mp_artifacts.artifacts example")

    # Build a tiny config so the saved snapshot matches a real study layout.
    config = get_base_config()
    config.study.name = "artifact_demo"
    config.outputs.run_name = "artifact_demo"

    # Write into a temporary study directory so the example leaves no residue.
    with tempfile.TemporaryDirectory() as tmpdir:
        study_dir = Path(tmpdir) / "demo_study"
        run_dir = create_run_layout(study_dir, config)

        # Save one example of each common artifact type.
        save_json(run_dir / "artifacts" / "demo.json", {"hello": "world"})
        save_tensor(run_dir / "artifacts" / "demo_tensor.pt", torch.arange(6))
        save_config_snapshot(run_dir, config)

        # Reload the latest run and config snapshot to check the round trip.
        latest = find_latest_run_dir(study_dir)
        reloaded = load_config_snapshot(run_dir)

        # These prints show the on-disk contract used by training and evaluation.
        print(f"run_dir              : {run_dir}")
        print(f"latest               : {latest}")
        print(f"saved demo json      : {run_dir / 'artifacts' / 'demo.json'}")
        print(f"saved demo tensor    : {run_dir / 'artifacts' / 'demo_tensor.pt'}")
        print(f"reloaded study name  : {reloaded.study.name}")
