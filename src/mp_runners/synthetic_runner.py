"""
Runner for controlled synthetic datasets with an MLP.

This is the canonical synthetic command-line workflow. It supports train-only,
train plus final evaluation, and replay evaluation from an existing finished
`DONE_` attempt. The runner owns synthetic CLI/config choices; shared training
stays in `mp_training.trainer`; synthetic metrics stay in `mp_metrics.metrics_synthetic`.
"""

import argparse
import os
from pathlib import Path
import sys

import torch

from gdfm import GDFM
from mp_artifacts import list_checkpoint_paths, load_attempt_status
from mp_config import clone_config, get_base_config
from mp_data import build_data_bundle
from mp_metrics.loss_eval import collect_loss_eval_record
from mp_metrics.metrics_synthetic import (
    collect_controlled_synthetic_metric_record,
    intrinsic_geometry_parts,
)
from mp_runners.common import (
    PARAMETERIZATION_CHOICES,
    artifact_paths,
    count_parameters,
)
from mp_runners.synthetic_eval import evaluate_synthetic_run
from mp_training import (
    CHECKPOINT_SCHEDULE_CHOICES,
    parse_checkpoint_steps,
    resolve_checkpoint_steps,
    train_run,
)
from mp_tracking import resolve_wandb_identity


RUNNER_NAME = "synthetic_runner"
DEFAULT_OUT_DIR = "outputs/runners/synthetic_runner"
DEFAULT_RUN_NAME = "synthetic_projected_gmm_mlp"
SYNTHETIC_DATASETS = ("gmm_subspace", "torus3d")
DONE_ATTEMPT_PREFIX = "DONE_"
EVAL_READY_STATES = {"trained", "completed"}
EVAL_REQUIRED_FILES = (
    "status.json",
    "artifacts/config.json",
    "artifacts/train_tensor.pt",
    "artifacts/reference_tensor.pt",
    "artifacts/z_init.pt",
    "artifacts/geometry.pt",
    "artifacts/run_summary.json",
)


def _default_wandb_entity(fallback):
    """Return the environment W&B entity or a fallback."""
    return os.getenv("WANDB_ENTITY", "").strip() or fallback


def default_settings():
    """Return canonical synthetic MLP defaults."""
    return {
        "device": "auto",
        "out_dir": DEFAULT_OUT_DIR,
        "run_name": DEFAULT_RUN_NAME,
        "seed": 0,
        "dataset": "gmm_subspace",
        "ambient_dim": 2,
        "intrinsic_data_dim": 2,
        # Legacy aliases kept for notebooks/scripts while the public CLI moves
        # to D/d language.
        "data_dim": 2,
        "intrinsic_dim": 2,
        "n_train": 10,
        "n_test": 1024,
        "n_clusters": 3,
        "sigma_z": 0.10,
        "sigma_eta": 1.0e-3,
        "geometry_seed": 0,
        "major_radius": 2.0,
        "minor_radius": 0.5,
        "hidden_dim": 256,
        "n_layers": 4,
        "time_emb_dim": 64,
        "prediction_space": "v",
        "loss_space": "v",
        "n_steps": 10000,
        "batch_cap": 10,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "log_every": 100,
        "ckpt_every": 2000,
        "ckpt_schedule": "bonnaire-1m",
        "ckpt_steps": "",
        "eval_samples": 256,
        "sampling_method": "heun",
        "sampling_steps": 128,
        "ratio_threshold": 0.3,
        "chunk_size": 512,
        "sliced_w2_projections": 128,
        "final_eval": False,
        "eval_run_dir": None,
        "wandb": False,
        "wandb_mode": "offline",
        "wandb_project": "",
        "wandb_entity": _default_wandb_entity("maj90"),
        "wandb_name": "",
        "wandb_group": "",
        "wandb_tags": "",
        "wandb_notes": "",
        "wandb_strict": False,
    }


def _as_settings(settings):
    """Return settings as a plain dictionary."""
    if settings is None:
        return default_settings()
    if isinstance(settings, argparse.Namespace):
        return vars(settings)
    return dict(settings)


def build_config(settings):
    """
    Build the canonical config for one controlled synthetic MLP run.

    The public settings use research-facing names: `ambient_dim` for emitted
    data dimension and `intrinsic_data_dim` for the known generator dimension.
    The config also keeps old aliases so notebooks can still read saved runs.
    """
    defaults = default_settings()
    provided = _as_settings(settings)
    settings = {**defaults, **provided}
    if (
        "data_dim" in provided
        and (
            "ambient_dim" not in provided
            or int(provided["ambient_dim"]) == int(defaults["ambient_dim"])
        )
    ):
        settings["ambient_dim"] = provided["data_dim"]
    if (
        "intrinsic_dim" in provided
        and (
            "intrinsic_data_dim" not in provided
            or int(provided["intrinsic_data_dim"])
            == int(defaults["intrinsic_data_dim"])
        )
    ):
        settings["intrinsic_data_dim"] = provided["intrinsic_dim"]
    settings["data_dim"] = settings["ambient_dim"]
    settings["intrinsic_dim"] = settings["intrinsic_data_dim"]
    dataset = str(settings["dataset"])
    if dataset not in SYNTHETIC_DATASETS:
        raise ValueError(f"Unknown synthetic dataset: {dataset}")

    config = clone_config(get_base_config())
    config.study.name = RUNNER_NAME
    config.study.description = "Controlled synthetic-data GDFM/MLP runner."
    config.study.dataset_key = dataset
    config.outputs.run_name = str(settings["run_name"])

    config.framework.family = "fm"
    config.framework.prediction_space = str(settings["prediction_space"])
    config.framework.loss_space = str(settings["loss_space"])

    config.data.dataset = dataset
    config.data.root = "synthetic"
    config.data.download = False
    config.data.data_source = "canonical"
    config.data.n_train = int(settings["n_train"])
    config.data.n_test = int(settings["n_test"])
    if dataset == "torus3d":
        ambient_dim = int(settings["ambient_dim"])
        if ambient_dim < 3:
            raise ValueError("torus3d requires --ambient-dim >= 3.")
        config.data.ambient_dim = ambient_dim
        # Torus has local manifold dimension 2, but no configurable global
        # linear intrinsic data dimension. Keep legacy fields equal to D.
        config.data.intrinsic_data_dim = ambient_dim
        config.data.data_dim = ambient_dim
        config.data.intrinsic_dim = ambient_dim
        config.data.base_embedding_dim = 3
        config.data.local_manifold_dim = 2
        config.data.metric_feature_dim = 4
    else:
        config.data.ambient_dim = int(settings["ambient_dim"])
        config.data.intrinsic_data_dim = int(settings["intrinsic_data_dim"])
        config.data.data_dim = int(settings["ambient_dim"])
        config.data.intrinsic_dim = int(settings["intrinsic_data_dim"])
    config.data.n_clusters = int(settings["n_clusters"])
    config.data.sigma_z = float(settings["sigma_z"])
    config.data.sigma_eta = float(settings["sigma_eta"])
    config.data.geometry_seed = int(settings["geometry_seed"])
    config.data.major_radius = float(settings["major_radius"])
    config.data.minor_radius = float(settings["minor_radius"])
    config.data.flatten = False

    config.network.arch = "mlp"
    config.network.hidden_dim = int(settings["hidden_dim"])
    config.network.n_layers = int(settings["n_layers"])
    config.network.time_emb_dim = int(settings["time_emb_dim"])

    config.training.seed = int(settings["seed"])
    config.training.n_training_steps = int(settings["n_steps"])
    config.training.batch_cap = int(settings["batch_cap"])
    config.training.batch_size = int(settings["batch_cap"])
    config.training.lr = float(settings["lr"])
    config.training.weight_decay = float(settings["weight_decay"])
    config.training.ckpt_every = int(settings["ckpt_every"])
    config.training.ckpt_schedule = str(settings["ckpt_schedule"])
    config.training.ckpt_steps = parse_checkpoint_steps(settings["ckpt_steps"])
    config.training.resolved_ckpt_steps = tuple(resolve_checkpoint_steps(config))
    config.training.log_every = int(settings["log_every"])
    config.training.optimizer = "adam"

    config.evaluation.n_samples = int(settings["eval_samples"])
    config.evaluation.batch_size = int(settings["batch_cap"])
    config.evaluation.ratio_threshold = float(settings["ratio_threshold"])
    config.evaluation.chunk_size = int(settings["chunk_size"])
    config.evaluation.sample_panel_count = min(8, int(settings["eval_samples"]))
    config.evaluation.loss_eval_times = (0.5, 0.75)
    config.evaluation.loss_eval_batch_size = int(settings["batch_cap"])

    config.sampling.method = str(settings["sampling_method"])
    config.sampling.n_steps = int(settings["sampling_steps"])
    config.sliced_w2_projections = int(settings["sliced_w2_projections"])

    config.tracking.wandb.enabled = bool(settings["wandb"])
    config.tracking.wandb.mode = str(settings["wandb_mode"])
    config.tracking.wandb.project = str(settings["wandb_project"])
    config.tracking.wandb.project_explicit = bool(
        settings.get("wandb_project_provided", False)
    )
    config.tracking.wandb.entity = str(settings["wandb_entity"])
    config.tracking.wandb.name = str(settings["wandb_name"])
    config.tracking.wandb.name_explicit = bool(
        settings.get("wandb_name_provided", False)
    )
    config.tracking.wandb.group = str(settings["wandb_group"])
    config.tracking.wandb.group_explicit = bool(
        settings.get("wandb_group_provided", False)
    )
    config.tracking.wandb.tags = [
        item.strip()
        for item in str(settings["wandb_tags"]).split(",")
        if item.strip()
    ]
    config.tracking.wandb.notes = str(settings["wandb_notes"])
    config.tracking.wandb.strict = bool(settings["wandb_strict"])
    resolve_wandb_identity(config)
    return config


def build_data(config):
    """Materialize the canonical controlled synthetic train/reference bundle."""
    return build_data_bundle(config)


def _resolve_device(device):
    """Resolve `auto` to an available torch device string."""
    if str(device) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_model(config, device):
    """Build the canonical GDFM wrapper and MLP backbone."""
    return GDFM(config).to(_resolve_device(device))


def _sliced_directions(config, intrinsic_dim):
    """Draw fixed random directions for intrinsic sliced-W2."""
    generator = torch.Generator().manual_seed(int(config.training.seed) + 1009)
    directions = torch.randn(
        int(config.sliced_w2_projections),
        int(intrinsic_dim),
        generator=generator,
    )
    return directions / directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)


def inspect_metrics(config, data_bundle, model=None, device="cpu"):
    """
    Compute controlled synthetic metrics and optional fixed/random-t loss.

    This evaluates reference samples against the training tensor in both
    ambient and intrinsic spaces. It is a quick way to inspect a data bundle
    before a full training run.
    """
    train = data_bundle["train"].cpu()
    reference = data_bundle["reference"].cpu()
    geometry = data_bundle["geometry"]
    train_intrinsic = intrinsic_geometry_parts(train, geometry, config=config)["intrinsic"]
    reference_intrinsic = intrinsic_geometry_parts(reference, geometry, config=config)["intrinsic"]
    count = min(int(config.evaluation.n_samples), int(reference.shape[0]))
    directions = _sliced_directions(config, int(train_intrinsic.shape[1]))
    result = {
        "data": _summarize_bundle(data_bundle),
        "reference_metrics": collect_controlled_synthetic_metric_record(
            step=0,
            samples=reference[:count],
            baseline_samples=reference[:count],
            train_tensor=train,
            train_intrinsic=train_intrinsic,
            geometry=geometry,
            reference_intrinsic=reference_intrinsic[:count],
            sliced_directions=directions,
            chunk_size=int(config.evaluation.chunk_size),
            config=config,
        ),
    }
    if model is not None:
        result["loss_eval"] = collect_loss_eval_record(
            0,
            model,
            reference,
            config,
            _resolve_device(device),
        )
    return result


def run_full_job(config, out_dir, device, evaluate=False):
    """
    Run canonical training, then synthetic checkpoint metrics if requested.

    Training saves the replayable attempt directory. The optional evaluation
    step reads the saved tensors, geometry, and fixed `z_init` from that
    directory.
    """
    device = _resolve_device(device)
    run_dir = train_run(config, Path(out_dir), device=device)
    evaluation = (
        evaluate_synthetic_run(run_dir, device=device)
        if evaluate
        else {
            "checkpoint_metrics": [],
            "loss_eval": [],
            "checkpoint_sample_previews": [],
            "figures": {},
            "final_metrics": {},
            "final_loss_eval": {},
        }
    )
    return {
        "run_dir": str(run_dir),
        "checkpoint_metrics": evaluation["checkpoint_metrics"],
        "loss_eval": evaluation["loss_eval"],
        "checkpoint_sample_previews": evaluation["checkpoint_sample_previews"],
        "figures": evaluation.get("figures", {}),
        "final_metrics": evaluation.get("final_metrics", {}),
        "final_loss_eval": evaluation.get("final_loss_eval", {}),
        "config": config,
        "batch": {
            "batch_cap": int(config.training.batch_cap),
            "effective_batch_size": min(
                int(config.training.batch_cap),
                int(config.data.n_train),
            ),
            "drop_last": int(config.data.n_train) > int(config.training.batch_cap),
        },
        "artifacts": artifact_paths(run_dir),
        "paths": {
            "status": str(Path(run_dir) / "status.json"),
            "train_history": str(Path(run_dir) / "metrics" / "train_history.json"),
            "metrics_along_training": str(Path(run_dir) / "metrics" / "metrics_along_training.json"),
            "checkpoint_metrics": str(Path(run_dir) / "metrics" / "checkpoint_metrics.json"),
            "loss_eval": str(Path(run_dir) / "metrics" / "loss_eval.json"),
            "checkpoint_sample_previews": str(Path(run_dir) / "metrics" / "checkpoint_sample_previews.json"),
            "final_samples": str(Path(run_dir) / "metrics" / "final_samples.pt"),
            "figures": str(Path(run_dir) / "figures"),
            "checkpoints": str(Path(run_dir) / "checkpoints"),
        },
    }


def _validate_eval_run_dir(run_dir):
    """Return a replayable DONE attempt directory or raise a clear error."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.name.startswith(DONE_ATTEMPT_PREFIX):
        raise ValueError(
            "eval-run-dir must be a finished DONE_ attempt directory; "
            f"got {run_dir}"
        )
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"eval-run-dir is missing status.json: {run_dir}")
    status = load_attempt_status(run_dir)
    state = str(status.get("state", ""))
    if state not in EVAL_READY_STATES:
        raise ValueError(
            "eval-run-dir must have status state 'trained' or 'completed'; "
            f"got {state!r} for {run_dir}"
        )
    missing = [
        str((run_dir / relative_path).resolve())
        for relative_path in EVAL_REQUIRED_FILES
        if not (run_dir / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "eval-run-dir is missing required saved-run artifacts: "
            + ", ".join(missing)
        )
    if not list_checkpoint_paths(run_dir):
        raise FileNotFoundError(f"eval-run-dir has no checkpoints: {run_dir / 'checkpoints'}")
    return run_dir


def run_eval_job(run_dir, device):
    """
    Replay synthetic checkpoint metrics for one finished DONE attempt.

    Replay uses saved artifacts rather than rebuilding the synthetic dataset
    from current defaults.
    """
    device = _resolve_device(device)
    run_dir = _validate_eval_run_dir(run_dir)
    evaluation = evaluate_synthetic_run(run_dir, device=device)
    return {
        "mode": "eval_run_dir",
        "run_dir": str(run_dir),
        "checkpoint_metrics": evaluation["checkpoint_metrics"],
        "loss_eval": evaluation["loss_eval"],
        "checkpoint_sample_previews": evaluation["checkpoint_sample_previews"],
        "figures": evaluation.get("figures", {}),
        "final_metrics": evaluation.get("final_metrics", {}),
        "final_loss_eval": evaluation.get("final_loss_eval", {}),
        "artifacts": artifact_paths(run_dir),
    }


def parse_args(argv=None):
    """Parse the synthetic MLP runner command line."""
    settings = default_settings()
    argv_list = sys.argv[1:] if argv is None else list(argv)
    ckpt_every_provided = _flag_present(argv_list, "--ckpt-every", "--ckpt_every")
    ckpt_steps_provided = _flag_present(argv_list, "--ckpt-steps", "--ckpt_steps")
    wandb_project_provided = _flag_present(argv_list, "--wandb-project", "--wandb_project")
    wandb_name_provided = _flag_present(argv_list, "--wandb-name", "--wandb_name")
    wandb_group_provided = _flag_present(argv_list, "--wandb-group", "--wandb_group")
    ambient_dim_provided = any(
        token == "--ambient-dim"
        or token.startswith("--ambient-dim=")
        or token == "--data-dim"
        or token.startswith("--data-dim=")
        for token in argv_list
    )
    intrinsic_data_dim_provided = any(
        token == "--intrinsic-data-dim"
        or token.startswith("--intrinsic-data-dim=")
        or token == "--intrinsic-dim"
        or token.startswith("--intrinsic-dim=")
        for token in argv_list
    )
    parser = argparse.ArgumentParser(
        description="Train a controlled synthetic GDFM MLP and replay synthetic metrics."
    )
    parser.add_argument("--device", default=settings["device"], help="auto, cuda, cuda:0, cpu, ...")
    parser.add_argument("--out-dir", default=settings["out_dir"])
    parser.add_argument("--run-name", default=settings["run_name"])
    parser.add_argument("--seed", type=int, default=settings["seed"])

    parser.add_argument("--dataset", choices=SYNTHETIC_DATASETS, default=settings["dataset"])
    parser.add_argument(
        "--ambient-dim",
        "--data-dim",
        dest="ambient_dim",
        type=int,
        default=settings["ambient_dim"],
        help="D: ambient emitted-data dimension for controlled synthetic vectors.",
    )
    parser.add_argument(
        "--intrinsic-data-dim",
        "--intrinsic-dim",
        dest="intrinsic_data_dim",
        type=int,
        default=settings["intrinsic_data_dim"],
        help="d: true intrinsic data dimension for controlled synthetic data.",
    )
    parser.add_argument("--n-train", type=int, default=settings["n_train"])
    parser.add_argument("--n-reference", "--n-eval", dest="n_test", type=int, default=settings["n_test"])
    parser.add_argument("--n-clusters", type=int, default=settings["n_clusters"])
    parser.add_argument("--sigma-z", type=float, default=settings["sigma_z"])
    parser.add_argument("--sigma-eta", type=float, default=settings["sigma_eta"])
    parser.add_argument("--geometry-seed", type=int, default=settings["geometry_seed"])
    parser.add_argument("--major-radius", type=float, default=settings["major_radius"])
    parser.add_argument("--minor-radius", type=float, default=settings["minor_radius"])

    parser.add_argument("--hidden-dim", type=int, default=settings["hidden_dim"])
    parser.add_argument("--n-layers", type=int, default=settings["n_layers"])
    parser.add_argument("--time-emb-dim", type=int, default=settings["time_emb_dim"])
    parser.add_argument("--prediction-space", choices=PARAMETERIZATION_CHOICES, default=settings["prediction_space"])
    parser.add_argument("--loss-space", choices=PARAMETERIZATION_CHOICES, default=settings["loss_space"])

    parser.add_argument("--n-steps", type=int, default=settings["n_steps"])
    parser.add_argument("--batch-cap", type=int, default=settings["batch_cap"])
    parser.add_argument("--lr", type=float, default=settings["lr"])
    parser.add_argument("--weight-decay", type=float, default=settings["weight_decay"])
    parser.add_argument("--log-every", type=int, default=settings["log_every"])
    parser.add_argument(
        "--ckpt-every",
        "--ckpt_every",
        type=int,
        default=settings["ckpt_every"],
    )
    parser.add_argument(
        "--ckpt-schedule",
        "--ckpt_schedule",
        choices=CHECKPOINT_SCHEDULE_CHOICES,
        default=settings["ckpt_schedule"],
        help="Checkpoint schedule. Explicit --ckpt-steps has priority.",
    )
    parser.add_argument(
        "--ckpt-steps",
        "--ckpt_steps",
        default=settings["ckpt_steps"],
        help="Comma-separated checkpoint steps. Step 0 and final step are always saved.",
    )

    parser.add_argument("--eval-samples", type=int, default=settings["eval_samples"])
    parser.add_argument("--sampling-method", choices=("euler", "heun"), default=settings["sampling_method"])
    parser.add_argument("--sampling-steps", type=int, default=settings["sampling_steps"])
    parser.add_argument("--ratio-threshold", type=float, default=settings["ratio_threshold"])
    parser.add_argument("--chunk-size", type=int, default=settings["chunk_size"])
    parser.add_argument("--sliced-w2-projections", type=int, default=settings["sliced_w2_projections"])
    parser.add_argument("--final-eval", action=argparse.BooleanOptionalAction, default=settings["final_eval"])
    parser.add_argument("--eval-run-dir", default=settings["eval_run_dir"])

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=settings["wandb"])
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default=settings["wandb_mode"])
    parser.add_argument("--wandb-project", default=settings["wandb_project"])
    parser.add_argument("--wandb-entity", default=settings["wandb_entity"])
    parser.add_argument("--wandb-name", default=settings["wandb_name"])
    parser.add_argument("--wandb-group", default=settings["wandb_group"])
    parser.add_argument("--wandb-tags", default=settings["wandb_tags"])
    parser.add_argument("--wandb-notes", default=settings["wandb_notes"])
    parser.add_argument("--wandb-strict", action="store_true", default=settings["wandb_strict"])
    args = parser.parse_args(argv_list)
    args.ckpt_every_provided = bool(ckpt_every_provided)
    args.ckpt_steps_provided = bool(ckpt_steps_provided)
    args.wandb_project_provided = bool(wandb_project_provided)
    args.wandb_name_provided = bool(wandb_name_provided)
    args.wandb_group_provided = bool(wandb_group_provided)
    if args.ckpt_every_provided and not args.ckpt_steps_provided:
        args.ckpt_schedule = "interval"
    args.ambient_dim_provided = bool(ambient_dim_provided)
    args.intrinsic_data_dim_provided = bool(intrinsic_data_dim_provided)
    if args.dataset == "torus3d" and not args.ambient_dim_provided:
        args.ambient_dim = 3
    # Compatibility attributes for old helper code and notebooks.
    args.data_dim = int(args.ambient_dim)
    args.intrinsic_dim = int(args.intrinsic_data_dim)
    return args


def _flag_present(argv, *names):
    """Return True when any command-line spelling appears in argv."""
    prefixes = tuple(f"{name}=" for name in names)
    return any(token in names or token.startswith(prefixes) for token in argv)


def _validate_args(args):
    """Validate synthetic runner command-line arguments."""
    checks = [
        (args.n_train >= 2, "--n-train must be at least 2 for nearest-neighbor metrics."),
        (args.n_test >= 1, "--n-reference/--n-eval must be at least 1."),
        (args.eval_samples >= 1, "--eval-samples must be at least 1."),
        (args.ambient_dim >= 1, "--ambient-dim must be at least 1."),
        (args.intrinsic_data_dim >= 1, "--intrinsic-data-dim must be at least 1."),
        (
            args.dataset != "torus3d" or args.ambient_dim >= 3,
            "torus3d requires --ambient-dim >= 3.",
        ),
        (
            args.dataset != "torus3d" or not bool(getattr(args, "intrinsic_data_dim_provided", False)),
            "torus3d has local_manifold_dim=2 but no configurable --intrinsic-data-dim.",
        ),
        (
            args.intrinsic_data_dim <= args.ambient_dim or args.dataset == "torus3d",
            "--intrinsic-data-dim must be <= --ambient-dim.",
        ),
        (args.n_clusters >= 1, "--n-clusters must be at least 1."),
        (args.sigma_z > 0.0, "--sigma-z must be positive."),
        (args.sigma_eta >= 0.0, "--sigma-eta must be non-negative."),
        (args.major_radius > 0.0, "--major-radius must be positive."),
        (args.minor_radius > 0.0, "--minor-radius must be positive."),
        (args.sliced_w2_projections >= 1, "--sliced-w2-projections must be at least 1."),
        (args.n_steps >= 1, "--n-steps must be at least 1."),
        (args.batch_cap >= 1, "--batch-cap must be at least 1."),
        (args.log_every >= 1, "--log-every must be at least 1."),
        (args.ckpt_every >= 1, "--ckpt-every must be at least 1."),
        (args.sampling_steps >= 1, "--sampling-steps must be at least 1."),
        (args.chunk_size >= 1, "--chunk-size must be at least 1."),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def _print_field(label, value):
    """Print one aligned terminal field."""
    print("{:<18} {}".format(label + ":", value))


def _print_result(result):
    """Print a concise train-run result."""
    _print_model_summary(result["config"])
    _print_field("run_dir", result["run_dir"])
    _print_field("checkpoints", result["artifacts"]["checkpoints"]["path"])
    _print_field("metrics", result["paths"]["checkpoint_metrics"])
    _print_field("loss_eval", result["paths"]["loss_eval"])
    if not result["checkpoint_metrics"]:
        _print_field("final_eval", "skipped; replay later with --eval-run-dir")


def _print_eval_result(result):
    """Print a concise eval-run result."""
    _print_field("eval_run_dir", result["run_dir"])
    _print_field("checkpoints", len(result["checkpoint_metrics"]))
    _print_field("metrics", result["artifacts"]["checkpoint_metrics"]["path"])
    _print_field("loss_eval", result["artifacts"]["loss_eval"]["path"])


def _print_model_summary(config):
    """Print model name and parameter counts."""
    model = build_model(config, "cpu")
    params = count_parameters(model)
    _print_field("model", "GDFM + MLP")
    _print_field(
        "model params",
        f"{params['total']:,} total | {params['trainable']:,} trainable",
    )


def main(argv=None):
    """CLI entrypoint for the controlled synthetic MLP runner."""
    args = parse_args(argv)
    if args.eval_run_dir is not None:
        result = run_eval_job(args.eval_run_dir, args.device)
        _print_eval_result(result)
        return result

    _validate_args(args)
    config = build_config(args)
    result = run_full_job(
        config,
        args.out_dir,
        args.device,
        evaluate=bool(args.final_eval),
    )
    _print_result(result)
    return result


def _summarize_bundle(bundle):
    """Return a compact data-bundle summary."""
    summary = {
        "train_shape": list(bundle["train"].shape),
        "reference_shape": list(bundle["reference"].shape),
        "train_source_id": bundle.get("train_source_id"),
        "reference_source_id": bundle.get("reference_source_id"),
    }
    if bundle.get("geometry") is not None:
        summary["geometry_keys"] = sorted(bundle["geometry"].keys())
    return summary


if __name__ == "__main__":
    main()
