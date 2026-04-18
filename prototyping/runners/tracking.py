"""Experiment tracking wrapper for prototype runners."""

from pathlib import Path


def scalar_payload(payload):
    return {key: value for key, value in payload.items() if value is not None}


class Tracker:
    def __init__(self, args, wandb_dir, config):
        self.args = args
        self.wandb_dir = Path(wandb_dir)
        self.config = config
        self.module = None
        self.run = None
        self.metadata = self._metadata(active=False)

        if not bool(args.wandb):
            return

        try:
            import wandb

            self.module = wandb
            self.run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity.strip() or None,
                mode=args.wandb_mode,
                dir=str(self.wandb_dir),
                name=args.wandb_name.strip() or None,
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
            "project": str(self.args.wandb_project),
            "entity": self.args.wandb_entity.strip() or None,
            "run_id": getattr(run, "id", None) if run is not None else None,
            "run_name": getattr(run, "name", None) if run is not None else None,
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
