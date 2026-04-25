"""
Runner for image datasets with the UNet adapter.

This is the canonical image command-line workflow. It supports two modes:

1. train-only: train a model and save checkpoints/artifacts;
2. data summary: materialize and summarize the selected image data.

The runner builds configs and prints summaries. Shared training behavior stays
in `mp_training.trainer`; image metrics stay in `mp_metrics.metrics_images`.
"""

import argparse
from pathlib import Path
import sys

from mp_metrics.loss_eval import collect_loss_eval_record
from mp_runners.image_eval import collect_reference_memory
from mp_runners.common import (
    PARAMETERIZATION_CHOICES,
    artifact_paths,
    count_parameters,
)
from mp_runners.image_config import (
    build_config as _build_config,
    get_default_settings as _default_settings,
    list_datasets as _list_datasets,
    list_image_sizes as _list_image_sizes,
    list_models as _list_models,
)

from gdfm import GDFM
from mp_data import build_data_bundle
from mp_runtime import PRECISION_CHOICES, configure_torch_runtime
from mp_training import CHECKPOINT_SCHEDULE_CHOICES, train_run

def default_settings():
    """Return the canonical image-runner default settings."""
    return _default_settings()


def build_config(settings):
    """Build the canonical config for one image-runner UNet run."""
    return _build_config(settings)


def build_data(config):
    """
    Materialize the canonical image train/reference bundle.

    The returned bundle contains dense train/reference tensors plus metadata
    such as labels, source indices, split names, and leakage-guard results.
    """
    return build_data_bundle(config)


def build_model(config, device):
    """Build the canonical GDFM wrapper and its UNet backbone."""
    configure_torch_runtime(config, device)
    return GDFM(config).to(device)


def inspect_metrics(config, data_bundle, model=None, device="cpu"):
    """
    Compute held-out memory and optional fixed/random-t loss diagnostics.

    This helper is useful before a full run: it checks the selected data bundle
    and, if a model is passed, runs the same loss probes used during evaluation.
    """
    train = data_bundle["train"].cpu()
    reference = data_bundle["reference"].cpu()
    result = {
        "data": _summarize_bundle(data_bundle),
        "reference_memory": collect_reference_memory(
            reference,
            train,
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
            device,
            fixed_times=tuple(config.evaluation.loss_eval_times),
        )
    return result


def summarize_data(config):
    """
    Materialize the data bundle and return a concise summary.

    This powers `--data_summary`, which checks data selection before training.
    """
    data_bundle = build_data(config)
    return _summarize_bundle(data_bundle)


def run_full_job(config, out_dir, device):
    """
    Run canonical image training.

    Training always saves a replayable attempt directory. Full checkpoint
    evaluation is intentionally outside this train runner; use
    `mp_runners.evaluate_checkpoints.evaluate_run(...)` after training.
    """
    # Training is shared; image-specific empirical metrics live outside this runner.
    run_dir = train_run(config, Path(out_dir), device=device)
    return {
        "run_dir": str(run_dir),
        "config": config,
        "artifacts": artifact_paths(run_dir),
    }


def _print_model_summary(config, model=None, model_param_summary=None):
    """Print concise model creation + parameterization metadata."""
    network = config.network
    framework = config.framework

    model_desc = f"GDFM + {str(network.arch).upper()} ({str(network.arch)})"
    topology = (
        f"base_ch={int(network.base_channels)} "
        f"channel_mult={tuple(network.channel_mult)} "
        f"attn={tuple(network.attn_resolutions)} "
        f"res_blocks={int(network.num_res_blocks)} "
        f"dropout={float(network.dropout)} "
        f"time_emb={network.time_emb_dim}"
    )
    _print_kv("model creation:", model_desc)
    _print_kv(
        "model parameterization:",
        f"pred={framework.prediction_space} loss={framework.loss_space}",
    )
    _print_kv("model topology:", topology)

    if model is None and model_param_summary is None:
        return
    if model is not None:
        params = count_parameters(model)
    else:
        params = dict(model_param_summary)
    _print_kv(
        "model params:",
        f"{params['total']:,} total | {params['trainable']:,} trainable",
    )


def parse_args(argv=None):
    """Parse the image runner command line."""
    settings = default_settings()
    argv_list = sys.argv[1:] if argv is None else list(argv)
    ckpt_every_provided = _flag_present(argv_list, "--ckpt_every")
    ckpt_steps_provided = _flag_present(argv_list, "--ckpt_steps")
    wandb_project_provided = _flag_present(argv_list, "--wandb_project")
    wandb_name_provided = _flag_present(argv_list, "--wandb_name")
    wandb_group_provided = _flag_present(argv_list, "--wandb_group")
    parser = argparse.ArgumentParser(description="Image UNet runner.")

    # Run identity.
    parser.add_argument("--device", default=settings["device"])
    parser.add_argument("--out_dir", default=settings["out_dir"])
    parser.add_argument("--run_name", default=settings["run_name"])
    parser.add_argument("--seed", type=int, default=settings["seed"])

    # Config selection.
    parser.add_argument(
        "--dataset",
        choices=tuple(_list_datasets()),
        default=settings["dataset"],
    )
    parser.add_argument(
        "--image_size",
        type=int,
        choices=tuple(_list_image_sizes()),
        default=settings["image_size"],
        help="Image resolution. The runner chooses the matching data/model profile.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(_list_models()),
        default=settings["model"],
        help="Image model family. The runner chooses the matching architecture defaults.",
    )
    # Objective and sweep settings.
    parser.add_argument(
        "--prediction_space",
        choices=PARAMETERIZATION_CHOICES,
        default=settings["prediction_space"],
    )
    parser.add_argument(
        "--loss_space",
        choices=PARAMETERIZATION_CHOICES,
        default=settings["loss_space"],
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=settings["n_train"],
        help="Number of real images used for training.",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=settings["n_test"],
        help="Number of real held-out/reference images materialized for later tests.",
    )
    parser.add_argument(
        "--n_training_steps",
        type=int,
        default=settings["n_training_steps"],
        help="Number of optimizer steps.",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=settings["ckpt_every"],
        help="Save a checkpoint every this many optimizer steps.",
    )
    parser.add_argument(
        "--ckpt_schedule",
        choices=CHECKPOINT_SCHEDULE_CHOICES,
        default=settings["ckpt_schedule"],
        help="Checkpoint schedule. Explicit --ckpt_steps has priority.",
    )
    parser.add_argument(
        "--ckpt_steps",
        default=settings["ckpt_steps"],
        help="Comma-separated checkpoint steps. Step 0 and final step are always saved.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=settings["log_every"],
        help="Print/log training loss every this many optimizer steps.",
    )
    parser.add_argument(
        "--batch_cap",
        type=int,
        default=settings["batch_cap"],
        help="Maximum minibatch size; smaller datasets use the dataset size.",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISION_CHOICES,
        default=settings["precision"],
        help="Numerical mode: fp32, CUDA TF32, or CUDA BF16 autocast.",
    )
    parser.add_argument(
        "--data_summary",
        action="store_true",
        default=settings["data_summary"],
        help="Materialize and summarize the selected data bundle, then exit.",
    )
    parser.add_argument(
        "--generation_monitor_samples",
        type=int,
        default=settings["generation_monitor_samples"],
        help="Number of fixed-z samples available to train-time image monitors.",
    )
    parser.add_argument(
        "--eval_sample_bank_size",
        type=int,
        default=settings["eval_sample_bank_size"],
        help="Fixed-z sample bank saved for post-hoc checkpoint evaluation.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=settings["sampling_steps"],
        help="Solver steps used to generate images from z_init.",
    )

    # Data location.
    parser.add_argument("--root", default=settings["root"])
    parser.add_argument(
        "--download",
        dest="download",
        action="store_true",
        default=settings["download"],
    )
    parser.add_argument("--no_download", dest="download", action="store_false")

    # W&B basics.
    parser.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        default=settings["wandb"],
    )
    parser.add_argument("--no_wandb", dest="wandb", action="store_false")
    parser.add_argument(
        "--wandb_mode",
        choices=("offline", "online"),
        default=settings["wandb_mode"],
    )
    parser.add_argument("--wandb_project", default=settings["wandb_project"])
    parser.add_argument("--wandb_name", default=settings["wandb_name"])
    parser.add_argument("--wandb_group", default=settings["wandb_group"])
    parser.add_argument("--wandb_tags", default=settings["wandb_tags"])
    parser.add_argument(
        "--compact_model_summary",
        dest="compact_model_summary",
        action="store_true",
        default=settings["compact_model_summary"],
        help="Print a compact model summary instead of full layer-wise parameters.",
    )
    parser.add_argument(
        "--no_compact_model_summary",
        dest="compact_model_summary",
        action="store_false",
    )

    # Fixed-noise generation monitor.
    parser.add_argument(
        "--generation_monitor",
        action="store_true",
        default=settings["generation_monitor"],
        help="Save/display train and generated-sample image grids and log them to W&B.",
    )
    parser.add_argument(
        "--generation_monitor_show",
        dest="generation_monitor_show",
        action="store_true",
        default=settings["generation_monitor_show"],
        help="Display generation monitor figures inline when running from a notebook.",
    )
    parser.add_argument(
        "--no_generation_monitor_show",
        dest="generation_monitor_show",
        action="store_false",
    )

    # Always-on train-time projected f_mem monitor.
    parser.add_argument(
        "--random_projected_f_mem_samples",
        type=int,
        default=settings["random_projected_f_mem_samples"],
        help="Fixed-z generated samples used by the projected f_mem monitor.",
    )
    parser.add_argument(
        "--random_projected_f_mem_resize",
        type=int,
        default=settings["random_projected_f_mem_resize"],
        help="Image size used before random projection in the projected f_mem monitor.",
    )
    parser.add_argument(
        "--random_projected_f_mem_proj_dim",
        type=int,
        default=settings["random_projected_f_mem_proj_dim"],
        help="Random projection dimension used by the projected f_mem monitor.",
    )
    parser.add_argument(
        "--random_projected_f_mem_proj_seed",
        type=int,
        default=settings["random_projected_f_mem_proj_seed"],
        help="Projection seed. Defaults to training seed plus a fixed offset.",
    )
    args = parser.parse_args(argv_list)
    args.ckpt_every_provided = bool(ckpt_every_provided)
    args.ckpt_steps_provided = bool(ckpt_steps_provided)
    args.wandb_project_provided = bool(wandb_project_provided)
    args.wandb_name_provided = bool(wandb_name_provided)
    args.wandb_group_provided = bool(wandb_group_provided)
    if args.ckpt_every_provided and not args.ckpt_steps_provided:
        args.ckpt_schedule = "interval"
    return args


def _flag_present(argv, *names):
    """Return True when any command-line spelling appears in argv."""
    prefixes = tuple(f"{name}=" for name in names)
    return any(token in names or token.startswith(prefixes) for token in argv)


def main(argv=None):
    """CLI entrypoint for the image UNet runner."""
    args = parse_args(argv)

    settings = vars(args)
    config = build_config(settings)

    if args.log_every is not None and args.log_every < 1:
        raise ValueError(f"log_every must be >= 1, got {args.log_every}")
    if args.data_summary:
        summary = summarize_data(config)
        _print_data_summary(config, summary)
        return summary

    result = run_full_job(config, args.out_dir, args.device)
    _print_result(result)
    return result["run_dir"]


def _print_kv(label, value):
    """Print one aligned terminal field."""
    print(f"{label:<18} {value}")


def _print_result(result):
    """Print a concise train-run result."""
    # Keep this summary easy to scan in a running terminal.
    _print_model_summary(
        result["config"],
        model_param_summary=result.get("model_param_summary"),
    )
    _print_kv("run_dir:", result["run_dir"])
    _print_kv("checkpoints:", result["artifacts"]["checkpoints"]["path"])
    _print_kv("figures:", result["artifacts"]["figures"]["path"])
    _print_kv("train_history:", result["artifacts"]["train_history"]["path"])


def _print_data_summary(config, summary):
    """Print the selected data-bundle summary."""
    _print_kv("dataset:", config.study.dataset_key)
    _print_kv("internal_dataset:", config.data.dataset)
    _print_kv("train_shape:", summary["train_shape"])
    _print_kv("reference_shape:", summary["reference_shape"])
    _print_kv("train_source:", summary["train_source_id"])
    _print_kv("reference_source:", summary["reference_source_id"])
    _print_kv("display_shape:", summary["display_shape"])
    _print_kv("train_indices:", summary["train_indices_preview"])
    _print_kv("reference_indices:", summary["reference_indices_preview"])


def _summarize_bundle(bundle):
    """Return a compact data-bundle summary."""
    summary = {
        "train_shape": list(bundle["train"].shape),
        "reference_shape": list(bundle["reference"].shape),
        "train_source_id": bundle.get("train_source_id"),
        "reference_source_id": bundle.get("reference_source_id"),
        "train_role": bundle.get("train_role"),
        "reference_role": bundle.get("reference_role"),
        "display_shape": bundle.get("display_shape"),
        "data_validation": bundle.get("data_validation"),
        "requested_balanced_train_classes": bundle.get(
            "requested_balanced_train_classes"
        ),
        "requested_balanced_reference_classes": bundle.get(
            "requested_balanced_reference_classes"
        ),
        "balanced_train_classes": bundle.get("balanced_train_classes"),
        "balanced_reference_classes": bundle.get("balanced_reference_classes"),
        "class_balance_adjustments": bundle.get("class_balance_adjustments"),
        "train_indices_preview": _preview_tensor(bundle.get("train_indices")),
        "reference_indices_preview": _preview_tensor(bundle.get("reference_indices")),
    }
    if bundle.get("train_class_counts") is not None:
        summary["train_class_counts"] = dict(bundle["train_class_counts"])
    if bundle.get("reference_class_counts") is not None:
        summary["reference_class_counts"] = dict(bundle["reference_class_counts"])
    if bundle.get("train_labels") is not None:
        summary["train_labels_shape"] = list(bundle["train_labels"].shape)
        summary["train_labels_preview"] = _preview_tensor(bundle["train_labels"])
    if bundle.get("reference_labels") is not None:
        summary["reference_labels_shape"] = list(bundle["reference_labels"].shape)
        summary["reference_labels_preview"] = _preview_tensor(bundle["reference_labels"])
    return summary


def _preview_tensor(tensor, limit=10):
    """Return a short integer preview for an optional tensor."""
    if tensor is None:
        return None
    return [int(item) for item in tensor.flatten()[:limit].tolist()]


if __name__ == "__main__":
    main()
