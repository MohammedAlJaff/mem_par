"""Runtime helpers for low-dimensional synthetic experiment modules."""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def _parse_yaml_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_yaml_value(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(item.strip()) for item in inner.split(",")]
    return _parse_yaml_scalar(stripped)


def _load_yaml_with_stdlib(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, sep, raw_value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line: {raw_line}")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        value = raw_value.strip()
        if value:
            current[key] = _parse_yaml_value(value)
        else:
            nested: dict[str, Any] = {}
            current[key] = nested
            stack.append((indent, nested))

    return root


def load_yaml(path: str | Path) -> dict[str, Any]:
    raw_text = Path(path).read_text()
    if yaml is not None:
        payload = yaml.safe_load(raw_text)
    else:
        payload = _load_yaml_with_stdlib(raw_text)
    return payload or {}


def load_paths(experiment_dir: Path, override: str | Path | None = None) -> tuple[dict[str, Any], Path | None]:
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"Paths file not found: {path}")
        return load_yaml(path), path

    for candidate in (experiment_dir / "paths.local.yaml", experiment_dir / "paths.example.yaml"):
        if candidate.exists():
            return load_yaml(candidate), candidate
    return {}, None


def load_config_and_paths(
    experiment_dir: Path,
    config_path: str | Path | None = None,
    paths_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    config = load_yaml(config_path or experiment_dir / "config.yaml")
    paths, resolved_paths_file = load_paths(experiment_dir, paths_path)
    return config, paths, resolved_paths_file


def _first_or_value(value: Any) -> Any:
    return value[0] if isinstance(value, list) else value


def resolve_run_config(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    fm_cfg = config.get("flow_matching", {})
    eval_cfg = config.get("evaluation", {})

    run_config: dict[str, Any] = {
        "pred_space": _first_or_value(fm_cfg.get("pred_space", "x")),
        "loss_space": _first_or_value(fm_cfg.get("loss_space", "v")),
        "seed": _first_or_value(eval_cfg.get("seeds", [0])),
    }

    for key in ("d", "n"):
        if key in data_cfg:
            run_config[key] = _first_or_value(data_cfg[key])

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                run_config[key] = value

    return run_config


def enumerate_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = config.get("data", {})
    fm_cfg = config.get("flow_matching", {})
    seeds = config.get("evaluation", {}).get("seeds", [0])

    axes: dict[str, list[Any]] = {
        "pred_space": fm_cfg.get("pred_space", ["x"]),
        "loss_space": fm_cfg.get("loss_space", ["v"]),
        "seed": seeds,
    }
    for key in ("d", "n"):
        if key in data_cfg:
            value = data_cfg[key]
            axes[key] = value if isinstance(value, list) else [value]

    normalized = {
        key: value if isinstance(value, list) else [value]
        for key, value in axes.items()
    }
    keys = list(normalized.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(normalized[key] for key in keys))]


def make_run_dir(experiment_dir: Path, run_config: dict[str, Any]) -> Path:
    parts = [f"pred{run_config['pred_space']}", f"loss{run_config['loss_space']}"]
    if "d" in run_config:
        parts.append(f"d{run_config['d']}")
    if "n" in run_config:
        parts.append(f"n{run_config['n']}")
    parts.append(f"seed{run_config['seed']}")

    run_dir = experiment_dir / "runs" / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def save_manifest(
    experiment_dir: Path,
    module_name: str,
    runs: list[dict[str, Any]],
) -> Path:
    manifest_path = experiment_dir / "runs" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "module": module_name,
        "experiment": experiment_dir.name,
        "runs": runs,
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def load_run_from_manifest(manifest_path: str | Path, run_index: int) -> dict[str, Any]:
    manifest = load_yaml(manifest_path) if str(manifest_path).endswith((".yaml", ".yml")) else json.loads(Path(manifest_path).read_text())
    return manifest["runs"][run_index]


def repo_root_from_experiment_dir(experiment_dir: Path) -> Path:
    return experiment_dir.resolve().parents[3]


def write_slurm_script(
    experiment_dir: Path,
    module_name: str,
    manifest_path: Path,
    n_runs: int,
    partition: str,
    time_limit: str,
    account: str,
    project_dir: str,
) -> Path:
    repo_root = repo_root_from_experiment_dir(experiment_dir)
    relative_experiment = experiment_dir.resolve().relative_to(repo_root)
    manifest_rel = manifest_path.resolve().relative_to(repo_root)
    log_pattern = Path(project_dir) / relative_experiment / "runs" / "slurm_%A_%a.out"

    script = f"""#!/bin/bash
#SBATCH --job-name={experiment_dir.name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={time_limit}
#SBATCH --output={log_pattern}
#SBATCH --error={log_pattern}
#SBATCH --array=0-{n_runs - 1}

module load Python/3.10.4-GCCcore-11.3.0
source /proj/{account}/envs/parambnd/bin/activate

cd {project_dir}
python -m {module_name} --run-index ${{SLURM_ARRAY_TASK_ID}} --manifest {manifest_rel}
"""

    script_path = experiment_dir / "runs" / "submit_slurm.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    return script_path


def submit_slurm(script_path: Path) -> None:
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    print(result.stdout.strip())


def execute_synthetic_run(
    experiment_dir: Path,
    config: dict[str, Any],
    run_config: dict[str, Any],
    build_dataset: Callable[[dict[str, Any], dict[str, Any]], Any],
    build_network: Callable[[dict[str, Any], dict[str, Any], Any], Any],
    device: str,
) -> dict[str, Any]:
    import torch

    from .evaluation.memorization import compute_f_mem, compute_nn_stats
    from .flow_matching import FlowMatching
    from .training.trainer import Trainer
    from .utils import (
        load_train_data,
        log_message,
        save_dataset_artifact,
        save_metrics,
        seed_everything,
    )

    seed_everything(run_config["seed"])

    run_dir = make_run_dir(experiment_dir, run_config)
    log_message(f"Run dir: {run_dir}", run_dir)
    log_message(f"Run config: {json.dumps(run_config, indent=2)}", run_dir)

    dataset = build_dataset(config, run_config)
    save_dataset_artifact(
        dataset,
        run_dir,
        {
            "family": "synthetic",
            "experiment": experiment_dir.name,
            **run_config,
        },
    )

    network = build_network(config, run_config, dataset)
    flow_model = FlowMatching(
        network=network,
        pred_space=run_config["pred_space"],
        loss_space=run_config["loss_space"],
    )

    trainer = Trainer(flow_model, dataset, config, str(run_dir), device=device)
    train_summary = trainer.train()

    eval_cfg = config.get("evaluation", {})
    flow_cfg = config.get("flow_matching", {})
    trainer.ema.apply()
    flow_model.eval()

    sample_item = dataset[0]
    n_gen = eval_cfg.get("n_gen_samples", 2000)
    sample_shape = (n_gen,) + tuple(sample_item.shape)
    with torch.no_grad():
        samples = flow_model.sample(
            shape=sample_shape,
            n_steps=eval_cfg.get("sample_steps", flow_cfg.get("sample_steps", 100)),
            method=flow_cfg.get("sampler", "heun"),
            device=device,
        ).cpu()
    trainer.ema.restore()
    torch.save(samples, run_dir / "samples" / "final_samples.pt")

    train_data = load_train_data(run_dir)
    nn_stats = compute_nn_stats(samples, train_data)
    f_mem = compute_f_mem(nn_stats["r"], threshold=eval_cfg.get("f_mem_threshold", 0.5))

    metrics = {
        "experiment": experiment_dir.name,
        **run_config,
        "f_mem": f_mem,
        "mean_d1": float(nn_stats["d1"].mean()),
        "mean_d2": float(nn_stats["d2"].mean()),
        "mean_r": float(nn_stats["r"].mean()),
        "median_r": float(nn_stats["r"].median()),
        **train_summary,
    }
    save_metrics(metrics, run_dir / "metrics.json")
    log_message(f"f_mem = {f_mem:.4f}", run_dir)
    return metrics


def collect_metrics(experiment_dir: Path):
    import pandas as pd

    rows = []
    for metrics_path in sorted((experiment_dir / "runs").rglob("metrics.json")):
        rows.append(json.loads(metrics_path.read_text()))
    return pd.DataFrame(rows)


def aggregate_over_seeds(df):
    group_cols = ["pred_space", "loss_space"]
    if "d" in df.columns:
        group_cols = ["d", *group_cols]
    if "n" in df.columns:
        group_cols = [*group_cols, "n"]

    metric_cols = [column for column in ("f_mem", "mean_d1", "mean_r", "median_r") if column in df.columns]
    agg_dict = {f"{column}_mean": (column, "mean") for column in metric_cols}
    agg_dict.update({f"{column}_std": (column, "std") for column in metric_cols})
    agg_dict["n_seeds"] = ("seed", "count")
    return df.groupby(group_cols).agg(**agg_dict).reset_index()


def fit_boundaries_per_condition(agg_df) -> dict[str, dict[str, Any]]:
    from .evaluation.memorization import fit_boundary

    results: dict[str, dict[str, Any]] = {}
    has_d = "d" in agg_df.columns
    group_cols = ["d", "pred_space", "loss_space"] if has_d else ["pred_space", "loss_space"]

    for keys, group in agg_df.groupby(group_cols):
        if has_d:
            d_value, pred_space, _loss_space = keys
        else:
            pred_space, _loss_space = keys
            d_value = "all"

        ordered = group.sort_values("n")
        boundary = fit_boundary(
            ordered["n"].to_numpy(),
            ordered["f_mem_mean"].to_numpy(),
            ordered["f_mem_std"].to_numpy() if "f_mem_std" in ordered else None,
        )

        bucket = str(d_value)
        results.setdefault(bucket, {})[pred_space] = boundary
    return results
