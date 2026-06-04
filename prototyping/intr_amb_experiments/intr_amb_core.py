"""
Sphere-only train/eval core for the intrinsic/ambient prototype experiments.

The target data distribution is a random embedded hypersphere shell. The source
distribution is always the clean standard normal N(0, I_D): radius and
off-manifold thickness affect only the target data.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gdfm import GDFM
from mp_artifacts import list_checkpoint_paths, load_config_snapshot, save_checkpoint
from mp_config import config_to_dict, get_base_config
from mp_metrics.metrics_synthetic import compute_memorization_metrics

from sphere_data import HypersphereShellDataset, PARAMS


DEFAULT_OUT_DIR = "prototyping/intr_amb_experiments/outputs/intrinsic_ambient_runs"
TARGET_DATASET = "intr_amb_hypersphere_shell"
SOURCE_DISTRIBUTION = "standard_normal"
SOURCE_COVARIANCE = "identity"
DEFAULT_WANDB_PROJECT = "memparam-intr-amb-hypersphere"

WANDB_GENERATED_FIELDS = (
    "f_mem",
    "mean_nn_ratio",
    "std_nn_ratio",
    "mean_d1",
    "mean_d2",
    "radius_mae",
    "off_manifold_dist_mean",
)

WANDB_GAP_FIELDS = (
    ("delta_f_mem", "f_mem"),
    ("delta_mean_nn_ratio", "mean_nn_ratio"),
    ("delta_radius_mae", "radius_mae"),
    ("delta_off_manifold_dist_mean", "off_manifold_dist_mean"),
)

WANDB_REFERENCE_FIELDS = (
    "baseline_f_mem",
    "baseline_mean_nn_ratio",
    "baseline_std_nn_ratio",
    "baseline_mean_d1",
    "baseline_mean_d2",
    "baseline_radius_mae",
    "baseline_off_manifold_dist_mean",
)

METRIC_FIELDS = (
    "param",
    "step",
    "loss",
    "f_mem",
    "mean_nn_ratio",
    "std_nn_ratio",
    "mean_d1",
    "mean_d2",
    "radius_mae",
    "off_manifold_dist_mean",
    "baseline_f_mem",
    "baseline_mean_nn_ratio",
    "baseline_std_nn_ratio",
    "baseline_mean_d1",
    "baseline_mean_d2",
    "baseline_radius_mae",
    "baseline_off_manifold_dist_mean",
    "delta_f_mem",
    "delta_mean_nn_ratio",
    "delta_radius_mae",
    "delta_off_manifold_dist_mean",
    "sample_count",
    "rho",
)


def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train one FM parameterization on hypersphere-shell prototype data."
    )
    parser.add_argument(
        "--param", required=True, choices=PARAMS, help="target parameterization"
    )
    parser.add_argument(
        "--D", "--ambient_dim", dest="D", type=int, default=8, help="ambient dimension"
    )
    parser.add_argument(
        "--K",
        "--intrinsic_dim",
        dest="K",
        type=int,
        default=2,
        help="intrinsic hypersphere dimension",
    )
    parser.add_argument(
        "--N",
        "--n_train",
        dest="N",
        type=int,
        default=100,
        help="number of train samples",
    )
    parser.add_argument(
        "--N_test",
        "--n_test",
        dest="N_test",
        type=int,
        default=1000,
        help="number of held-out reference samples",
    )
    parser.add_argument(
        "--radius",
        choices=("unit", "sqrtD"),
        default="unit",
        help="target hypersphere radius convention",
    )
    parser.add_argument(
        "--off_manifold_sigma",
        type=float,
        default=0.05,
        help="target-data off-manifold thickness; source noise remains N(0, I_D)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--geometry_seed", type=int, default=0)
    parser.add_argument(
        "--tau",
        "--steps",
        dest="tau",
        type=int,
        default=50000,
        help="number of optimizer steps",
    )
    parser.add_argument(
        "--B", "--batch_size", dest="B", type=int, default=64, help="minibatch size"
    )
    parser.add_argument(
        "--H", "--hidden_dim", dest="H", type=int, default=256, help="MLP hidden width"
    )
    parser.add_argument(
        "--L",
        "--n_layers",
        dest="L",
        type=int,
        default=4,
        help="number of MLP hidden layers",
    )
    parser.add_argument(
        "--T_emb",
        "--time_emb_dim",
        dest="T_emb",
        type=int,
        default=64,
        help="time embedding dimension",
    )
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--rho", type=float, default=0.3, help="NN-ratio memorization threshold"
    )
    parser.add_argument(
        "--N_gen",
        "--sample_count",
        dest="N_gen",
        type=int,
        default=1000,
        help="number of generated samples per checkpoint",
    )
    parser.add_argument(
        "--tau_sample",
        "--sampling_steps",
        dest="tau_sample",
        type=int,
        default=128,
        help="ODE solver steps for generation",
    )
    parser.add_argument("--sampling_method", choices=("euler", "heun"), default="heun")
    parser.add_argument(
        "--tau_eval",
        "--ckpt_every",
        dest="tau_eval",
        type=int,
        default=5000,
        help="checkpoint/evaluation interval in optimizer steps",
    )
    parser.add_argument(
        "--tau_ckpts",
        "--ckpt_steps",
        dest="tau_ckpts",
        default="",
        help="explicit comma-separated checkpoint steps",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--print_eval", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        default=False,
        help="log config and metrics to Weights & Biases",
    )
    parser.add_argument(
        "--no_wandb",
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_mode",
        "--wandb-mode",
        dest="wandb_mode",
        choices=("offline", "online"),
        default="online",
        help="Weights & Biases mode",
    )
    parser.add_argument(
        "--wandb_project",
        "--wandb-project",
        dest="wandb_project",
        default="",
        help=f"Weights & Biases project; default: {DEFAULT_WANDB_PROJECT}",
    )
    parser.add_argument(
        "--wandb_entity",
        "--wandb-entity",
        dest="wandb_entity",
        default=os.getenv("WANDB_ENTITY", "").strip(),
        help="Weights & Biases entity/team",
    )
    parser.add_argument(
        "--wandb_name",
        "--wandb-name",
        dest="wandb_name",
        default="",
        help="W&B run name",
    )
    parser.add_argument(
        "--wandb_group",
        "--wandb-group",
        dest="wandb_group",
        default="",
        help="W&B run group",
    )
    parser.add_argument(
        "--wandb_tags",
        "--wandb-tags",
        dest="wandb_tags",
        default="",
        help="comma-separated extra W&B tags",
    )
    parser.add_argument(
        "--wandb_notes",
        "--wandb-notes",
        dest="wandb_notes",
        default="",
        help="W&B notes",
    )
    parser.add_argument(
        "--wandb_strict",
        "--wandb-strict",
        dest="wandb_strict",
        action="store_true",
        help="fail the run if W&B initialization or logging fails",
    )
    return parser.parse_args(sys.argv[1:] if argv is None else list(argv))


def parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay hypersphere prototype checkpoint evaluation."
    )
    parser.add_argument("--run_dir", required=True, help="saved prototype run directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--print_eval", action="store_true")
    return parser.parse_args(sys.argv[1:] if argv is None else list(argv))


def resolve_device(name):
    if str(name) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(name)


def _resolved_radius(args):
    if str(args.radius) == "sqrtD":
        return math.sqrt(float(args.D))
    return 1.0


def validate_train_args(args):
    checks = [
        (args.D >= 1, "--D must be at least 1."),
        (args.K >= 1, "--K must be at least 1."),
        (args.K + 1 <= args.D, "--K must satisfy K + 1 <= --D."),
        (args.N >= 2, "--N must be at least 2."),
        (args.N_test >= 2, "--N_test must be at least 2."),
        (args.off_manifold_sigma >= 0.0, "--off_manifold_sigma must be non-negative."),
        (args.tau >= 1, "--tau must be at least 1."),
        (args.B >= 1, "--B must be at least 1."),
        (args.N_gen >= 1, "--N_gen must be at least 1."),
        (args.tau_sample >= 1, "--tau_sample must be at least 1."),
        (args.tau_eval >= 1, "--tau_eval must be at least 1."),
        (args.chunk_size >= 1, "--chunk_size must be at least 1."),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def checkpoint_steps(args):
    if str(args.tau_ckpts).strip():
        steps = {
            int(item.strip()) for item in str(args.tau_ckpts).split(",") if item.strip()
        }
    else:
        steps = set(range(0, int(args.tau) + 1, int(args.tau_eval)))
    steps.add(0)
    steps.add(int(args.tau))
    return tuple(sorted(step for step in steps if 0 <= int(step) <= int(args.tau)))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_history(history, step, loss):
    if history and int(history[-1]["step"]) == int(step):
        return False
    history.append({"step": int(step), "loss": float(loss)})
    return True


def append_jsonl(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in METRIC_FIELDS})


def write_metric_rows(run_dir, rows):
    csv_path = Path(run_dir) / "metrics" / "checkpoint_metrics.csv"
    jsonl_path = Path(run_dir) / "metrics" / "checkpoint_metrics.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in METRIC_FIELDS})
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def make_optimizer(config, model):
    if str(config.training.optimizer) == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
            momentum=float(config.training.momentum),
        )
    if str(config.training.optimizer) == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )
    raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def sample_model(model, z_init, config, device):
    outputs = []
    batch_size = int(config.evaluation.batch_size)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, z_init.shape[0], batch_size):
                z = z_init[start : start + batch_size].to(device)
                samples = model.sample(
                    z,
                    n_steps=int(config.sampling.n_steps),
                    method=str(config.sampling.method),
                    device=device,
                )
                outputs.append(samples.detach().cpu().float())
    finally:
        if was_training:
            model.train()
    return torch.cat(outputs, dim=0)


def sphere_quality_metrics(samples, dataset, chunk_size):
    radius_error_sum = 0.0
    off_manifold_sum = 0.0
    total_count = 0

    with torch.no_grad():
        for start in range(0, samples.shape[0], int(chunk_size)):
            x = samples[start : start + int(chunk_size)].to(
                dataset.device, dataset.dtype
            )
            radius_error_sum += float(dataset.projected_radius_error(x).sum().item())
            off_manifold_sum += float(dataset.dist_to_manifold(x).sum().item())
            total_count += int(x.shape[0])

    total_count = max(total_count, 1)
    return {
        "radius_mae": radius_error_sum / total_count,
        "off_manifold_dist_mean": off_manifold_sum / total_count,
    }


def build_dataset(args):
    dataset = HypersphereShellDataset(
        d=int(args.D),
        k=int(args.K),
        radius=_resolved_radius(args),
        off_manifold_sigma=float(args.off_manifold_sigma),
        n_train=int(args.N),
        seed=int(args.geometry_seed),
        dtype=torch.float64,
    )
    train = dataset.train_set.detach().cpu().float()
    reference_seed = int(args.geometry_seed) + 100003
    reference = dataset.sample(int(args.N_test), seed=reference_seed).detach().cpu().float()
    if train.shape == reference.shape and torch.equal(train, reference):
        raise ValueError("train and reference tensors are exactly equal.")
    return dataset, train, reference


def build_dataset_from_config(config):
    return HypersphereShellDataset(
        d=int(config.data.ambient_dim),
        k=int(config.data.intrinsic_dim),
        radius=float(config.data.radius),
        off_manifold_sigma=float(config.data.off_manifold_sigma),
        n_train=int(config.data.n_train),
        seed=int(config.data.geometry_seed),
        dtype=torch.float64,
    )


def data_meta(args, train, reference):
    return {
        "dataset": TARGET_DATASET,
        "dataset_family": "sphere",
        "target_distribution": "hypersphere_shell",
        "train_shape": list(train.shape),
        "reference_shape": list(reference.shape),
        "param": str(args.param),
        "D": int(args.D),
        "K": int(args.K),
        "subspace_dim": int(args.K) + 1,
        "N": int(args.N),
        "N_test": int(args.N_test),
        "radius": _resolved_radius(args),
        "radius_mode": str(args.radius),
        "off_manifold_sigma": float(args.off_manifold_sigma),
        "seed": int(args.seed),
        "geometry_seed": int(args.geometry_seed),
        "optimizer": str(args.optimizer),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "momentum": float(args.momentum),
        "source_distribution": SOURCE_DISTRIBUTION,
        "source_covariance": SOURCE_COVARIANCE,
        "source_dim": int(args.D),
    }


def _auto_run_name(args):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        f"sphere_D{args.D}_K{args.K}_R{args.radius}_N{args.N}_"
        f"{args.param}_tau{args.tau}_seed{args.seed}_gseed{args.geometry_seed}_{stamp}"
    )


def _wandb_slug(value, fallback):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text).strip("-_")
    return text or fallback


def _wandb_project(args):
    project = str(args.wandb_project or "").strip()
    return project or DEFAULT_WANDB_PROJECT


def _wandb_name(args):
    name = str(args.wandb_name or "").strip()
    return name or str(args.run_id or _auto_run_name(args))


def _wandb_group(args):
    group = str(args.wandb_group or "").strip()
    if group:
        return group
    return (
        f"sphere-D{args.D}-K{args.K}-R{args.radius}-"
        f"N{args.N}-tau{args.tau}"
    )


def _wandb_tags(args):
    tags = [
        "intr-amb",
        "hypersphere",
        "source-standard-normal",
        f"param-{args.param}",
        f"radius-{args.radius}",
    ]
    tags.extend(
        item.strip()
        for item in str(args.wandb_tags or "").split(",")
        if item.strip()
    )
    return tags


def _wandb_run_id(args):
    return _wandb_slug(args.run_id or _wandb_name(args), "intr-amb-sphere-run")


def make_run_dir(args):
    root = Path(args.out_dir).expanduser()
    if args.run_id:
        run_dir = root / args.run_id
    else:
        base_name = _auto_run_name(args)
        name = base_name
        run_dir = root / name
        suffix = 2
        while run_dir.exists():
            name = f"{base_name}_{suffix}"
            run_dir = root / name
            suffix += 1
        args.run_id = name
    for subdir in ("artifacts", "checkpoints", "metrics", "figures", "logs"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=False)
    return run_dir


def build_config(args):
    config = get_base_config()
    config.study.name = "intr_amb_hypersphere_single"
    config.study.description = (
        "Hypersphere-shell target with standard-normal source prototype."
    )
    config.study.dataset_key = TARGET_DATASET
    config.outputs.run_name = str(args.run_id or "auto")
    config.framework.family = "fm"
    config.framework.prediction_space = str(args.param)
    config.framework.loss_space = str(args.param)
    config.data.dataset = TARGET_DATASET
    config.data.data_dim = int(args.D)
    config.data.ambient_dim = int(args.D)
    config.data.intrinsic_dim = int(args.K)
    config.data.intrinsic_data_dim = int(args.K)
    config.data.subspace_dim = int(args.K) + 1
    config.data.n_train = int(args.N)
    config.data.n_test = int(args.N_test)
    config.data.geometry_seed = int(args.geometry_seed)
    config.data.radius = _resolved_radius(args)
    config.data.radius_mode = str(args.radius)
    config.data.off_manifold_sigma = float(args.off_manifold_sigma)
    config.network.arch = "mlp"
    config.network.hidden_dim = int(args.H)
    config.network.n_layers = int(args.L)
    config.network.time_emb_dim = int(args.T_emb)
    config.training.seed = int(args.seed)
    config.training.n_training_steps = int(args.tau)
    config.training.batch_cap = int(args.B)
    config.training.batch_size = int(args.B)
    config.training.lr = float(args.lr)
    config.training.weight_decay = float(args.weight_decay)
    config.training.optimizer = str(args.optimizer)
    config.training.momentum = float(args.momentum)
    config.training.ckpt_every = int(args.tau_eval)
    config.training.ckpt_steps = tuple(checkpoint_steps(args))
    config.training.resolved_ckpt_steps = tuple(checkpoint_steps(args))
    config.evaluation.sample_bank_size = int(args.N_gen)
    config.evaluation.n_samples = int(args.N_gen)
    config.evaluation.batch_size = int(args.B)
    config.evaluation.ratio_threshold = float(args.rho)
    config.evaluation.chunk_size = int(args.chunk_size)
    config.sampling.method = str(args.sampling_method)
    config.sampling.n_steps = int(args.tau_sample)
    config.source_distribution = SOURCE_DISTRIBUTION
    config.source_covariance = SOURCE_COVARIANCE
    config.tracking.wandb.enabled = bool(args.wandb)
    config.tracking.wandb.mode = str(args.wandb_mode)
    config.tracking.wandb.project = _wandb_project(args)
    config.tracking.wandb.entity = str(args.wandb_entity or "")
    config.tracking.wandb.name = _wandb_name(args)
    config.tracking.wandb.group = _wandb_group(args)
    config.tracking.wandb.tags = _wandb_tags(args)
    config.tracking.wandb.notes = str(args.wandb_notes or "")
    config.tracking.wandb.strict = bool(args.wandb_strict)
    config.tracking.wandb.dir = "logs/wandb"
    return config


def _wandb_config_payload(config, run_dir):
    return {
        "run/param": str(config.framework.prediction_space),
        "run/dataset": str(config.data.dataset),
        "run/run_dir": str(Path(run_dir).resolve()),
        "data/ambient_dim": int(config.data.ambient_dim),
        "data/intrinsic_dim": int(config.data.intrinsic_dim),
        "data/subspace_dim": int(config.data.subspace_dim),
        "data/n_train": int(config.data.n_train),
        "data/n_test": int(config.data.n_test),
        "data/radius_mode": str(config.data.radius_mode),
        "data/radius": float(config.data.radius),
        "data/off_manifold_sigma": float(config.data.off_manifold_sigma),
        "data/geometry_seed": int(config.data.geometry_seed),
        "source/distribution": str(config.source_distribution),
        "source/covariance": str(config.source_covariance),
        "source/dim": int(config.data.ambient_dim),
        "model/arch": str(config.network.arch),
        "model/hidden_dim": int(config.network.hidden_dim),
        "model/n_layers": int(config.network.n_layers),
        "model/time_emb_dim": int(config.network.time_emb_dim),
        "train/steps": int(config.training.n_training_steps),
        "train/batch_size": int(config.training.batch_size),
        "train/optimizer": str(config.training.optimizer),
        "train/lr": float(config.training.lr),
        "train/weight_decay": float(config.training.weight_decay),
        "train/momentum": float(config.training.momentum),
        "train/seed": int(config.training.seed),
        "eval/n_gen": int(config.evaluation.n_samples),
        "eval/rho": float(config.evaluation.ratio_threshold),
        "eval/chunk_size": int(config.evaluation.chunk_size),
        "eval/checkpoint_every": int(config.training.ckpt_every),
        "eval/checkpoint_steps": [
            int(step) for step in tuple(config.training.resolved_ckpt_steps)
        ],
        "sampling/method": str(config.sampling.method),
        "sampling/n_steps": int(config.sampling.n_steps),
    }


def _prepare_wandb_dirs(wandb_dir):
    dirs = {
        "WANDB_DIR": wandb_dir,
        "WANDB_CACHE_DIR": wandb_dir / "cache",
        "WANDB_CONFIG_DIR": wandb_dir / "config",
        "WANDB_DATA_DIR": wandb_dir / "data",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for key, path in dirs.items():
        os.environ.setdefault(key, str(path.resolve()))
    return {key.lower(): os.environ.get(key) for key in dirs}


def init_wandb(args, config, run_dir):
    if not bool(args.wandb):
        return None

    wandb_dir = Path(run_dir) / "logs" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_paths = _prepare_wandb_dirs(wandb_dir)
    metadata_path = Path(run_dir) / "artifacts" / "wandb.json"
    project = _wandb_project(args)
    entity = str(args.wandb_entity or "").strip() or None
    run_name = _wandb_name(args)
    run_group = _wandb_group(args)

    try:
        wandb_module = importlib.import_module("wandb")
        run = wandb_module.init(
            project=project,
            entity=entity,
            dir=str(wandb_dir.resolve()),
            mode=str(args.wandb_mode),
            id=_wandb_run_id(args),
            resume="allow",
            name=run_name,
            group=run_group,
            job_type=str(config.study.name),
            notes=str(args.wandb_notes or "").strip() or None,
            tags=_wandb_tags(args),
            config=_wandb_config_payload(config, run_dir),
            reinit=True,
        )
        metadata = {
            "enabled": True,
            "active": True,
            "mode": str(args.wandb_mode),
            "project": project,
            "entity": entity,
            "run_id": getattr(run, "id", _wandb_run_id(args)),
            "run_name": getattr(run, "name", run_name),
            "group": run_group,
            "run_url": getattr(run, "url", None),
            "dir": str(wandb_dir.resolve()),
            "paths": wandb_paths,
        }
        save_json(metadata_path, metadata)
        print(
            "wandb active "
            f"mode={metadata['mode']} project={project} "
            f"name={metadata['run_name']} group={run_group}",
            flush=True,
        )
        return run
    except Exception as exc:
        metadata = {
            "enabled": True,
            "active": False,
            "mode": str(args.wandb_mode),
            "project": project,
            "entity": entity,
            "run_name": run_name,
            "group": run_group,
            "warning": str(exc),
            "dir": str(wandb_dir.resolve()),
            "paths": wandb_paths,
        }
        save_json(metadata_path, metadata)
        if bool(args.wandb_strict):
            raise
        print(f"wandb skipped: {exc}", flush=True)
        return None


def log_wandb_metrics(wandb_run, args, metrics, prefix, step):
    if wandb_run is None:
        return

    payload = {}
    if str(prefix) == "train":
        value = metrics.get("loss")
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                payload["train/loss"] = number
    elif str(prefix) == "eval":
        for key in WANDB_GENERATED_FIELDS:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                number = float(value)
                if math.isfinite(number):
                    payload[f"generated/{key}"] = number
        for source_key, target_key in WANDB_GAP_FIELDS:
            value = metrics.get(source_key)
            if isinstance(value, (int, float)):
                number = float(value)
                if math.isfinite(number):
                    payload[f"gap/{target_key}"] = number
    if not payload:
        return

    try:
        wandb_run.log(payload, step=int(step))
    except Exception as exc:
        if bool(args.wandb_strict):
            raise
        print(f"wandb metric logging skipped: {exc}", flush=True)


def _wandb_reference_summary_key(key):
    return f"reference/{str(key)[len('baseline_'):]}"


def log_wandb_reference_summary(wandb_run, args, metrics):
    if wandb_run is None:
        return

    payload = {}
    for key in WANDB_REFERENCE_FIELDS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                payload[_wandb_reference_summary_key(key)] = number
    for source_key, target_key in (
        ("sample_count", "run/sample_count"),
        ("rho", "run/rho"),
    ):
        value = metrics.get(source_key)
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                payload[target_key] = number
    if not payload:
        return

    try:
        for key, value in payload.items():
            wandb_run.summary[key] = value
    except Exception as exc:
        if bool(args.wandb_strict):
            raise
        print(f"wandb reference summary logging skipped: {exc}", flush=True)


def log_wandb_run_summary(wandb_run, args, final_loss, final_step):
    if wandb_run is None:
        return

    payload = {"run/final_step": float(final_step)}
    if isinstance(final_loss, (int, float)):
        number = float(final_loss)
        if math.isfinite(number):
            payload["run/final_loss"] = number

    try:
        for key, value in payload.items():
            wandb_run.summary[key] = value
    except Exception as exc:
        if bool(args.wandb_strict):
            raise
        print(f"wandb run summary logging skipped: {exc}", flush=True)


def finish_wandb(wandb_run, args):
    if wandb_run is None:
        return
    try:
        wandb_run.finish()
    except Exception as exc:
        if bool(args.wandb_strict):
            raise
        print(f"wandb finish skipped: {exc}", flush=True)


def _prefixed(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def print_eval_summary(row):
    loss_text = "nan" if row["loss"] is None else f"{float(row['loss']):.6g}"
    print(
        "eval "
        f"param={row['param']} step={row['step']} loss={loss_text} "
        f"f_mem={row['f_mem']:.6g} baseline_f_mem={row['baseline_f_mem']:.6g} "
        f"delta_f_mem={row['delta_f_mem']:.6g} "
        f"mean_r={row['mean_nn_ratio']:.6g} "
        f"baseline_mean_r={row['baseline_mean_nn_ratio']:.6g} "
        f"delta_mean_r={row['delta_mean_nn_ratio']:.6g} "
        f"radius_mae={row['radius_mae']:.6g} "
        f"baseline_radius_mae={row['baseline_radius_mae']:.6g} "
        f"off_manifold_dist_mean={row['off_manifold_dist_mean']:.6g} "
        f"baseline_off_manifold_dist_mean={row['baseline_off_manifold_dist_mean']:.6g}",
        flush=True,
    )


def evaluate_samples(samples, step, loss, train, reference, dataset, args):
    generated = compute_memorization_metrics(
        samples,
        train,
        ratio_threshold=float(args.rho),
        chunk_size=int(args.chunk_size),
    )
    generated.update(sphere_quality_metrics(samples, dataset, args.chunk_size))

    baseline_count = min(int(args.N_gen), int(reference.shape[0]))
    baseline_samples = reference[:baseline_count]
    baseline = compute_memorization_metrics(
        baseline_samples,
        train,
        ratio_threshold=float(args.rho),
        chunk_size=int(args.chunk_size),
    )
    baseline.update(sphere_quality_metrics(baseline_samples, dataset, args.chunk_size))

    row = {
        "param": str(args.param),
        "step": int(step),
        "loss": None if loss is None else float(loss),
        "sample_count": int(samples.shape[0]),
        "rho": float(args.rho),
        **generated,
        **_prefixed("baseline", baseline),
    }
    row["delta_f_mem"] = row["f_mem"] - row["baseline_f_mem"]
    row["delta_mean_nn_ratio"] = row["mean_nn_ratio"] - row["baseline_mean_nn_ratio"]
    row["delta_radius_mae"] = row["radius_mae"] - row["baseline_radius_mae"]
    row["delta_off_manifold_dist_mean"] = (
        row["off_manifold_dist_mean"] - row["baseline_off_manifold_dist_mean"]
    )
    return row


def evaluate_checkpoint(
    model, step, loss, train, reference, z_init, dataset, config, args, run_dir, device
):
    samples = sample_model(model, z_init, config, device)
    row = evaluate_samples(samples, step, loss, train, reference, dataset, args)
    append_csv(run_dir / "metrics" / "checkpoint_metrics.csv", row)
    append_jsonl(run_dir / "metrics" / "checkpoint_metrics.jsonl", row)
    if args.print_eval:
        print_eval_summary(row)
    return row


def record_sampling_trajectory(model, z_init, config, device, max_frames=80):
    n_steps = int(config.sampling.n_steps)
    method = str(config.sampling.method)
    frame_stride = max(1, n_steps // int(max_frames))
    batch_size = int(z_init.shape[0])
    was_training = model.training
    model.eval()

    with torch.no_grad():
        z = z_init.to(device)
        ts = torch.linspace(model.t_min, model.t_max, n_steps + 1, device=device)
        frames = [(0, float(ts[0].item()), z.detach().cpu().float())]

        try:
            for index in range(len(ts) - 1):
                dt = ts[index + 1] - ts[index]
                if method == "euler":
                    velocity = model._velocity(z, ts[index].item(), batch_size)
                    z = z + dt * velocity
                elif method == "heun":
                    velocity_1 = model._velocity(z, ts[index].item(), batch_size)
                    z_tilde = z + dt * velocity_1
                    velocity_2 = model._velocity(
                        z_tilde, ts[index + 1].item(), batch_size
                    )
                    z = z + 0.5 * dt * (velocity_1 + velocity_2)
                else:
                    raise ValueError(f"Unknown sampling method: {method}")

                frame_step = index + 1
                if frame_step % frame_stride == 0 or frame_step == n_steps:
                    frames.append(
                        (frame_step, float(ts[frame_step].item()), z.detach().cpu().float())
                    )
        finally:
            if was_training:
                model.train()

    return frames


def axis_limits_2d(*tensors):
    points = torch.cat(
        [tensor.detach().cpu().float()[:, :2] for tensor in tensors], dim=0
    )
    lo = points.min(dim=0).values
    hi = points.max(dim=0).values
    center = 0.5 * (lo + hi)
    radius = 0.55 * float((hi - lo).max().item())
    radius = max(radius, 1.0e-3)
    return (
        float(center[0] - radius),
        float(center[0] + radius),
        float(center[1] - radius),
        float(center[1] + radius),
    )


def axis_limits_3d(*tensors):
    points = torch.cat(
        [tensor.detach().cpu().float()[:, :3] for tensor in tensors], dim=0
    )
    lo = points.min(dim=0).values
    hi = points.max(dim=0).values
    center = 0.5 * (lo + hi)
    radius = 0.55 * float((hi - lo).max().item())
    radius = max(radius, 1.0e-3)
    return (
        float(center[0] - radius),
        float(center[0] + radius),
        float(center[1] - radius),
        float(center[1] + radius),
        float(center[2] - radius),
        float(center[2] + radius),
    )


def _setup_matplotlib(run_dir):
    os.environ.setdefault("XDG_CACHE_HOME", str((run_dir / "logs" / "cache").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str((run_dir / "logs" / "mpl").resolve()))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def save_final_2d_visualizations(model, train, z_init, config, args, run_dir, device):
    if int(config.data.ambient_dim) != 2:
        return

    plt = _setup_matplotlib(run_dir)
    import matplotlib.animation as animation

    generated = sample_model(model, z_init, config, device)
    x_min, x_max, y_min, y_max = axis_limits_2d(z_init, generated, train)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(train[:, 0], train[:, 1], s=18, c="black", alpha=0.30, label="train", linewidths=0)
    ax.scatter(z_init[:, 0], z_init[:, 1], s=14, c="#4c78a8", alpha=0.12, label="source z_init", linewidths=0)
    ax.scatter(generated[:, 0], generated[:, 1], s=14, c="#f58518", alpha=0.65, label="generated", linewidths=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"final 2D samples: param={args.param}, sampler={config.sampling.method}"
    )
    ax.grid(alpha=0.20)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "final_2d_source_generated_train.png", dpi=180)
    plt.close(fig)

    if not animation.writers.is_available("ffmpeg"):
        print("2D video skipped: matplotlib ffmpeg writer is not available.", flush=True)
        return

    frames = record_sampling_trajectory(model, z_init, config, device)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(train[:, 0], train[:, 1], s=18, c="black", alpha=0.22, label="train", linewidths=0)
    particles = ax.scatter(frames[0][2][:, 0], frames[0][2][:, 1], s=14, c="#f58518", alpha=0.70, label="samples", linewidths=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.20)
    ax.legend(loc="best")

    def update(frame):
        frame_step, t_value, points = frame
        particles.set_offsets(points[:, :2])
        ax.set_title(
            f"sampling: param={args.param}, sampler={config.sampling.method}, "
            f"step={frame_step}/{config.sampling.n_steps}, t={t_value:.3f}"
        )
        return (particles,)

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=80,
        blit=False,
        repeat=False,
    )
    try:
        writer = animation.FFMpegWriter(fps=12, bitrate=1800)
        movie.save(run_dir / "figures" / "final_2d_sampling_process.mp4", writer=writer)
    except Exception as exc:
        print(f"2D video skipped: failed to write MP4 ({exc}).", flush=True)
    finally:
        plt.close(fig)


def save_final_3d_visualizations(model, train, z_init, config, args, run_dir, device):
    if int(config.data.ambient_dim) != 3:
        return

    plt = _setup_matplotlib(run_dir)
    import matplotlib.animation as animation

    generated = sample_model(model, z_init, config, device)
    x_min, x_max, y_min, y_max, z_min, z_max = axis_limits_3d(
        z_init, generated, train
    )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(train[:, 0], train[:, 1], train[:, 2], s=16, c="black", alpha=0.22, label="train", depthshade=False, linewidths=0)
    ax.scatter(z_init[:, 0], z_init[:, 1], z_init[:, 2], s=12, c="#4c78a8", alpha=0.10, label="source z_init", depthshade=False, linewidths=0)
    ax.scatter(generated[:, 0], generated[:, 1], generated[:, 2], s=12, c="#f58518", alpha=0.65, label="generated", depthshade=False, linewidths=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_title(
        f"final 3D samples: param={args.param}, sampler={config.sampling.method}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "final_3d_source_generated_train.png", dpi=180)
    plt.close(fig)

    if not animation.writers.is_available("ffmpeg"):
        print("3D video skipped: matplotlib ffmpeg writer is not available.", flush=True)
        return

    frames = record_sampling_trajectory(model, z_init, config, device)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(train[:, 0], train[:, 1], train[:, 2], s=16, c="black", alpha=0.16, label="train", depthshade=False, linewidths=0)
    particles = ax.scatter(frames[0][2][:, 0], frames[0][2][:, 1], frames[0][2][:, 2], s=12, c="#f58518", alpha=0.70, label="samples", depthshade=False, linewidths=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.legend(loc="best")

    def update(frame):
        frame_step, t_value, points = frame
        particles._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        ax.view_init(elev=22.0, azim=45.0 + 0.45 * frame_step)
        ax.set_title(
            f"sampling: param={args.param}, sampler={config.sampling.method}, "
            f"step={frame_step}/{config.sampling.n_steps}, t={t_value:.3f}"
        )
        return (particles,)

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=80,
        blit=False,
        repeat=False,
    )
    try:
        writer = animation.FFMpegWriter(fps=12, bitrate=1800)
        movie.save(run_dir / "figures" / "final_3d_sampling_process.mp4", writer=writer)
    except Exception as exc:
        print(f"3D video skipped: failed to write MP4 ({exc}).", flush=True)
    finally:
        plt.close(fig)


def save_final_visualizations(model, train, z_init, config, args, run_dir, device):
    ambient_dim = int(config.data.ambient_dim)
    if ambient_dim == 2:
        save_final_2d_visualizations(model, train, z_init, config, args, run_dir, device)
        return
    if ambient_dim == 3:
        save_final_3d_visualizations(model, train, z_init, config, args, run_dir, device)


def save_figures(rows, run_dir):
    plt = _setup_matplotlib(run_dir)
    steps = [int(row["step"]) for row in rows]
    figure_specs = [
        (
            "f_mem_over_steps.png",
            "f_mem",
            "f_mem",
            [float(row["baseline_f_mem"]) for row in rows],
        ),
        (
            "mean_nn_ratio_over_steps.png",
            "mean_nn_ratio",
            "mean d1/d2 (lower = closer to train)",
            [float(row["baseline_mean_nn_ratio"]) for row in rows],
        ),
        (
            "radius_mae_over_steps.png",
            "radius_mae",
            "projected radius MAE (lower = closer to shell)",
            [float(row["baseline_radius_mae"]) for row in rows],
        ),
        (
            "off_manifold_dist_mean_over_steps.png",
            "off_manifold_dist_mean",
            "mean off-manifold distance",
            [float(row["baseline_off_manifold_dist_mean"]) for row in rows],
        ),
    ]
    for filename, key, ylabel, baseline in figure_specs:
        values = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, values, marker="o", linewidth=1.8, label="generated")
        ax.plot(steps, baseline, linestyle="--", linewidth=1.4, label="held-out")
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / filename, dpi=160)
        plt.close(fig)

    delta_specs = [
        (
            "delta_f_mem_over_steps.png",
            "delta_f_mem",
            "generated f_mem - held-out f_mem",
        ),
        (
            "delta_mean_nn_ratio_over_steps.png",
            "delta_mean_nn_ratio",
            "generated mean d1/d2 - held-out mean d1/d2",
        ),
        (
            "delta_radius_mae_over_steps.png",
            "delta_radius_mae",
            "generated radius MAE - held-out radius MAE",
        ),
        (
            "delta_off_manifold_dist_mean_over_steps.png",
            "delta_off_manifold_dist_mean",
            "generated off-manifold distance - held-out off-manifold distance",
        ),
    ]
    for filename, key, ylabel in delta_specs:
        values = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.plot(steps, values, marker="o", linewidth=1.8)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / filename, dpi=160)
        plt.close(fig)


def train_one(args):
    validate_train_args(args)
    device = resolve_device(args.device)

    torch.manual_seed(int(args.seed))

    run_dir = make_run_dir(args)
    config = build_config(args)
    dataset, train, reference = build_dataset(args)

    z_gen = torch.Generator().manual_seed(int(args.seed) + 2003)
    z_init = torch.randn(int(args.N_gen), int(args.D), generator=z_gen)

    torch.save(train, run_dir / "artifacts" / "train_tensor.pt")
    torch.save(reference, run_dir / "artifacts" / "reference_tensor.pt")
    torch.save(z_init, run_dir / "artifacts" / "z_init.pt")
    save_json(run_dir / "artifacts" / "config.json", config_to_dict(config))
    save_json(run_dir / "artifacts" / "data_meta.json", data_meta(args, train, reference))
    wandb_run = init_wandb(args, config, run_dir)

    model = GDFM(config).to(device)
    optimizer = make_optimizer(config, model)
    ckpts = set(checkpoint_steps(args))
    history = []
    last_loss = None

    save_checkpoint(run_dir, 0, model, optimizer)
    step0_row = evaluate_checkpoint(
        model,
        0,
        None,
        train,
        reference,
        z_init,
        dataset,
        config,
        args,
        run_dir,
        device,
    )
    metric_rows = [step0_row]
    log_wandb_reference_summary(wandb_run, args, step0_row)
    log_wandb_metrics(wandb_run, args, step0_row, "eval", 0)

    batch_gen = torch.Generator().manual_seed(int(args.seed) + 17)
    progress = range(1, int(args.tau) + 1)
    if not args.no_progress:
        from tqdm.auto import tqdm

        progress = tqdm(progress, desc=f"train:sphere:{args.param}", unit="step")

    for step in progress:
        idx = torch.randint(
            0,
            train.shape[0],
            (min(int(args.B), train.shape[0]),),
            generator=batch_gen,
        )
        batch = train[idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        _append_history(history, step, last_loss)
        log_wandb_metrics(wandb_run, args, {"loss": last_loss}, "train", step)
        if not args.no_progress:
            progress.set_postfix(loss=f"{last_loss:.4f}", refresh=False)

        if int(step) in ckpts:
            save_json(run_dir / "metrics" / "train_history.json", history)
            save_checkpoint(run_dir, step, model, optimizer)
            row = evaluate_checkpoint(
                model,
                step,
                last_loss,
                train,
                reference,
                z_init,
                dataset,
                config,
                args,
                run_dir,
                device,
            )
            metric_rows.append(row)
            log_wandb_metrics(wandb_run, args, row, "eval", step)

    save_json(run_dir / "metrics" / "train_history.json", history)
    save_figures(metric_rows, run_dir)
    save_final_visualizations(model, train, z_init, config, args, run_dir, device)
    log_wandb_run_summary(wandb_run, args, last_loss, int(args.tau))
    save_json(
        run_dir / "artifacts" / "run_summary.json",
        {
            "run_dir": str(run_dir.resolve()),
            "dataset": "sphere",
            "dataset_key": TARGET_DATASET,
            "param": str(args.param),
            "tau": int(args.tau),
            "checkpoint_steps": sorted(int(step) for step in ckpts),
            "final_loss": last_loss,
            "optimizer": str(args.optimizer),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "momentum": float(args.momentum),
            "source_distribution": SOURCE_DISTRIBUTION,
            "source_covariance": SOURCE_COVARIANCE,
            "wandb_enabled": bool(args.wandb),
            "wandb_mode": str(args.wandb_mode),
            "wandb_project": _wandb_project(args),
            "wandb_name": _wandb_name(args),
            "wandb_group": _wandb_group(args),
            "metrics_path": str(
                (run_dir / "metrics" / "checkpoint_metrics.csv").resolve()
            ),
            "figures_dir": str((run_dir / "figures").resolve()),
        },
    )
    finish_wandb(wandb_run, args)
    return run_dir


def replay_run(run_dir, device="cpu", print_eval=False):
    run_dir = Path(run_dir)
    config = load_config_snapshot(run_dir)
    dataset_name = str(config.data.dataset)
    if dataset_name != TARGET_DATASET:
        raise ValueError(
            f"Run dataset is {dataset_name}, but this evaluator expects {TARGET_DATASET}."
        )
    device = resolve_device(device)

    train = torch.load(run_dir / "artifacts" / "train_tensor.pt", map_location="cpu")
    reference = torch.load(
        run_dir / "artifacts" / "reference_tensor.pt", map_location="cpu"
    )
    z_init = torch.load(run_dir / "artifacts" / "z_init.pt", map_location="cpu")
    dataset = build_dataset_from_config(config)
    history_path = run_dir / "metrics" / "train_history.json"
    history = []
    if history_path.exists():
        history = json.loads(history_path.read_text())
    history_by_step = {int(item["step"]): float(item["loss"]) for item in history}

    args = argparse.Namespace(
        param=str(config.framework.prediction_space),
        D=int(config.data.ambient_dim),
        K=int(config.data.intrinsic_dim),
        N=int(config.data.n_train),
        N_test=int(config.data.n_test),
        radius=str(getattr(config.data, "radius_mode", "unit")),
        off_manifold_sigma=float(config.data.off_manifold_sigma),
        seed=int(config.training.seed),
        geometry_seed=int(config.data.geometry_seed),
        tau=int(config.training.n_training_steps),
        B=int(config.training.batch_cap),
        H=int(config.network.hidden_dim),
        L=int(config.network.n_layers),
        T_emb=int(config.network.time_emb_dim),
        optimizer=str(config.training.optimizer),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
        momentum=float(config.training.momentum),
        rho=float(config.evaluation.ratio_threshold),
        N_gen=int(config.evaluation.n_samples),
        tau_sample=int(config.sampling.n_steps),
        sampling_method=str(config.sampling.method),
        tau_eval=int(config.training.ckpt_every),
        tau_ckpts="",
        chunk_size=int(config.evaluation.chunk_size),
        device=device,
        out_dir="",
        run_id=str(config.outputs.run_name),
        print_eval=bool(print_eval),
        no_progress=True,
    )

    model = GDFM(config).to(device)
    checkpoint_paths = list_checkpoint_paths(run_dir)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

    rows = []
    for checkpoint_path in checkpoint_paths:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        step = int(state["step"])
        loss = history_by_step.get(step)
        row = evaluate_samples(
            sample_model(model, z_init, config, device),
            step,
            loss,
            train,
            reference,
            dataset,
            args,
        )
        rows.append(row)
        if print_eval:
            print_eval_summary(row)

    write_metric_rows(run_dir, rows)
    save_figures(rows, run_dir)
    save_final_visualizations(model, train, z_init, config, args, run_dir, device)
    return run_dir


def main(argv=None):
    args = parse_train_args(argv)
    run_dir = train_one(args)
    print(f"run_dir: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
