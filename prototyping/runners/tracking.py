"""Experiment tracking wrapper for prototype runners."""

import os
import re
from pathlib import Path


def scalar_payload(payload):
    return {key: value for key, value in payload.items() if value is not None}


def _slug(value, fallback):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-")
    return text or fallback


def _config_value(config, section, field, fallback):
    value = config.get(section, {}).get(field, fallback)
    return value if value not in (None, "") else fallback


def _default_wandb_project(config):
    project = os.getenv("WANDB_PROJECT", "").strip()
    if project:
        return project
    dataset = _slug(_config_value(config, "data", "dataset", "dataset"), "dataset")
    model = _slug(_config_value(config, "network", "arch", "model"), "model")
    return f"memparam-{dataset}-{model}"


def _default_wandb_run_name(config):
    dataset = _slug(_config_value(config, "data", "dataset", "dataset"), "dataset")
    model = _slug(_config_value(config, "network", "arch", "model"), "model")
    n_train = int(_config_value(config, "data", "n_train", 0))
    prediction = _slug(
        _config_value(config, "framework", "prediction_space", "pred"),
        "pred",
    )
    loss = _slug(_config_value(config, "framework", "loss_space", "loss"), "loss")
    seed = int(_config_value(config, "training", "seed", 0))
    return f"{dataset}-{model}-N{n_train}-{prediction}_{loss}-s{seed}"


class Tracker:
    def __init__(self, args, wandb_dir, config):
        self.args = args
        self.wandb_dir = Path(wandb_dir)
        self.config = config
        self.module = None
        self.run = None
        self.project = args.wandb_project.strip() or _default_wandb_project(config)
        self.run_name = args.wandb_name.strip() or _default_wandb_run_name(config)
        self.metadata = self._metadata(active=False)

        if not bool(args.wandb):
            return

        try:
            import wandb

            self.module = wandb
            self.run = wandb.init(
                project=self.project,
                entity=args.wandb_entity.strip() or None,
                mode=args.wandb_mode,
                dir=str(self.wandb_dir),
                name=self.run_name,
                group=args.wandb_group.strip() or None,
                tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
                notes=args.wandb_notes.strip() or None,
                config=config,
            )
            self.metadata = self._metadata(active=True)
        except Exception as exc:
            self.metadata = self._metadata(
                active=False,
                warning="W&B init failed: {}: {}".format(type(exc).__name__, exc),
            )
            if args.wandb_strict:
                raise

    def _metadata(self, *, active, warning=None):
        run = self.run
        return {
            "enabled": bool(self.args.wandb),
            "active": bool(active),
            "finished": False,
            "mode": str(self.args.wandb_mode),
            "project": str(self.project),
            "entity": self.args.wandb_entity.strip() or None,
            "run_id": getattr(run, "id", None) if run is not None else None,
            "run_name": getattr(run, "name", self.run_name) if run is not None else None,
            "run_path": getattr(run, "path", None) if run is not None else None,
            "run_url": getattr(run, "url", None) if run is not None else None,
            "dir": str(self.wandb_dir.resolve()),
            "warning": warning,
        }

    def log_scalars(self, payload, step=None):
        if self.run is not None:
            self.run.log(scalar_payload(payload), step=step)

    def log_images(self, figure_paths, step=None):
        if self.run is None or self.module is None:
            return
        payload = {}
        for name, path in figure_paths.items():
            suffix = Path(path).suffix.lower()
            if suffix in (".png", ".jpg", ".jpeg"):
                payload["figures/{}".format(name)] = self.module.Image(path)
        if payload:
            self.run.log(payload, step=step)

    def finish(self):
        if self.run is None:
            return self.metadata
        self.run.finish()
        self.metadata = dict(self.metadata)
        self.metadata["active"] = False
        self.metadata["finished"] = True
        return self.metadata
