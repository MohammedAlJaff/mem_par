"""Small persistence helpers for prototype runners."""

import json
from datetime import datetime
from pathlib import Path


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def make_run_dirs(out_dir, run_name, seed):
    out_root = Path(out_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "{}_seed{}_{}".format(run_name, seed, timestamp)
    run_dir = out_root / stem
    suffix = 1
    while run_dir.exists():
        run_dir = out_root / "{}_{:02d}".format(stem, suffix)
        suffix += 1

    paths = {
        "run": run_dir,
        "artifacts": run_dir / "artifacts",
        "checkpoints": run_dir / "checkpoints",
        "metrics": run_dir / "metrics",
        "figures": run_dir / "figures",
        "logs": run_dir / "logs",
        "wandb": run_dir / "logs" / "wandb",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return out_root, paths, timestamp


def update_latest_links(out_root, links):
    warnings = []
    for name, target in links.items():
        link_path = Path(out_root) / name
        try:
            if link_path.is_symlink():
                link_path.unlink()
            elif link_path.exists():
                warnings.append("{} exists and is not a symlink".format(link_path))
                continue
            link_path.symlink_to(Path(target), target_is_directory=True)
        except OSError as exc:
            warnings.append("{}: {}".format(link_path, exc))
    return warnings


def save_tensors(torch, tensors):
    for path, tensor in tensors.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
