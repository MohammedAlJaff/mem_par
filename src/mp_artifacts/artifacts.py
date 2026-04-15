import json
import re
from datetime import datetime
from pathlib import Path

import torch
from ml_collections import ConfigDict

from mp_config import config_to_dict


def _slug(text):
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text)).strip("-")


def make_run_id(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _slug(config.outputs.run_name)
    return (
        f"{base}_pred-{_slug(config.framework.prediction_space)}"
        f"_loss-{_slug(config.framework.loss_space)}"
        f"_seed-{int(config.training.seed)}_{timestamp}"
    )


def create_run_layout(study_dir, config):
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
    latest_file = Path(study_dir) / "runs" / "LATEST"
    if not latest_file.exists():
        raise FileNotFoundError(f"No latest run marker found at {latest_file}")
    run_id = latest_file.read_text().strip()
    run_dir = Path(study_dir) / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Latest run directory not found: {run_dir}")
    return run_dir


def resolve_run_dir(study_dir, run_dir=None):
    if run_dir is not None:
        return Path(run_dir)
    return find_latest_run_dir(study_dir)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())


def save_tensor(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def save_config_snapshot(run_dir, config):
    payload = config_to_dict(config)
    save_json(Path(run_dir) / "artifacts" / "config.json", payload)


def load_config_snapshot(run_dir):
    payload = load_json(Path(run_dir) / "artifacts" / "config.json")
    return ConfigDict(payload)


def save_checkpoint(run_dir, step, model, optimizer):
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
    return sorted((Path(run_dir) / "checkpoints").glob("step_*.pt"))
