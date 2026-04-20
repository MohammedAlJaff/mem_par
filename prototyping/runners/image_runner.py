"""
Readable prototype recipe for image datasets with the UNet adapter.
"""

import argparse
from pathlib import Path
import sys
import time

# Direct script execution from prototyping/runners should still find repo packages.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from prototyping.runners.common import (
    PARAMETERIZATION_CHOICES,
    artifact_paths,
    count_parameters,
)
from prototyping.runners.config_presets import list_presets
from prototyping.runners.metrics_image32 import (
    collect_loss_eval_record,
    collect_reference_memory,
    evaluate_image32_run,
)
from prototyping.runners.image_runner_config_builder import (
    build_config as _build_config,
    get_default_settings as _default_settings,
    list_datasets as _list_datasets,
)

from gdfm import GDFM
from mp_config import clone_config
from mp_data import build_data_bundle
from mp_runtime import (
    PRECISION_CHOICES,
    autocast_context,
    configure_torch_runtime,
    torch_device,
    validate_precision,
)
from mp_training import train_run


def default_settings():
    """Return the canonical image-runner default settings for this recipe."""
    return _default_settings()


def build_config(settings):
    """Build the canonical config for one image-runner UNet run."""
    return _build_config(settings)


def build_data(config):
    """Materialize the canonical image train/reference bundle."""
    return build_data_bundle(config)


def build_model(config, device):
    """Build the canonical GDFM wrapper and its UNet backbone."""
    configure_torch_runtime(config, device)
    return GDFM(config).to(device)


def inspect_metrics(config, data_bundle, model=None, device="cpu"):
    """Compute held-out memory and optional fixed/random-t loss diagnostics."""
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


def run_full_job(config, out_dir, device, evaluate=True):
    """Run canonical training, then image checkpoint metrics if requested."""
    # Training remains canonical; empirical metrics live beside this runner.
    run_dir = train_run(config, Path(out_dir), device=device)
    evaluation = (
        evaluate_image32_run(
            run_dir,
            device=device,
            fixed_times=tuple(config.evaluation.loss_eval_times),
        )
        if evaluate
        else {
            "checkpoint_metrics": [],
            "loss_eval": [],
            "checkpoint_sample_previews": [],
        }
    )
    return {
        "run_dir": str(run_dir),
        "checkpoint_metrics": evaluation["checkpoint_metrics"],
        "loss_eval": evaluation["loss_eval"],
        "checkpoint_sample_previews": evaluation["checkpoint_sample_previews"],
        "config": config,
        "artifacts": artifact_paths(run_dir),
    }


def compare_numerics(config, device):
    """
    Run a tiny in-memory training benchmark for fp32 vs a faster CUDA mode.

    This intentionally avoids W&B and the run-artifact layout. It uses the
    configured dataset, model, optimizer, batch cap, and number of steps so the
    table answers the practical question: "what changes when I flip precision?"
    """
    selected_precision = str(config.runtime.precision)
    target_precision = "bf16" if selected_precision == "fp32" else selected_precision
    precisions = ("fp32", target_precision)
    for precision in precisions:
        trial_config = clone_config(config)
        trial_config.runtime.precision = precision
        validate_precision(trial_config, device)

    data_bundle = build_data(config)
    train_tensor = data_bundle["train"].cpu()
    results = []
    for precision in precisions:
        trial_config = clone_config(config)
        trial_config.runtime.precision = precision
        trial_config.tracking.wandb.enabled = False
        runtime = configure_torch_runtime(trial_config, device)

        seed = int(trial_config.training.seed)
        torch.manual_seed(seed)
        torch_dev = torch_device(device)
        if torch_dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch_dev)

        model = build_model(trial_config, device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(trial_config.training.lr),
            weight_decay=float(trial_config.training.weight_decay),
        )

        batch_cap = int(trial_config.training.batch_cap)
        batch_size = min(batch_cap, int(train_tensor.shape[0]))
        drop_last = int(train_tensor.shape[0]) > batch_cap
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            train_tensor,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            generator=generator,
            pin_memory=bool(runtime["pin_memory"]),
        )
        iterator = iter(loader)
        n_steps = int(trial_config.training.n_steps)
        first_loss = None
        final_loss = None

        if torch_dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(torch_dev)
        start = time.perf_counter()
        model.train()
        for _ in range(n_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = batch.to(device, non_blocking=bool(runtime["non_blocking_h2d"]))

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(trial_config, device):
                loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            if first_loss is None:
                first_loss = loss_value
            final_loss = loss_value

        if torch_dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(torch_dev)
            peak_memory_gb = torch.cuda.max_memory_allocated(torch_dev) / 1.0e9
        else:
            peak_memory_gb = 0.0
        elapsed = time.perf_counter() - start

        results.append(
            {
                "precision": precision,
                "avg_step_time_sec": elapsed / max(1, n_steps),
                "peak_cuda_memory_gb": peak_memory_gb,
                "first_loss": first_loss,
                "final_loss": final_loss,
                **runtime,
            }
        )

        del model, optimizer
        if torch_dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "mode": "compare_numerics",
        "device": str(device),
        "n_steps": int(config.training.n_steps),
        "n_train": int(config.data.n_train),
        "batch_cap": int(config.training.batch_cap),
        "results": results,
    }
    _print_numerics_compare(result)
    return result


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
    parser = argparse.ArgumentParser(description="Image UNet prototype recipe.")

    # Run identity.
    parser.add_argument("--device", default=settings["device"])
    parser.add_argument("--out-dir", default=settings["out_dir"])
    parser.add_argument("--run-name", default=settings["run_name"])
    parser.add_argument("--seed", type=int, default=settings["seed"])

    # Config selection.
    parser.add_argument(
        "--dataset",
        choices=tuple(_list_datasets()),
        default=settings["dataset"],
    )
    parser.add_argument(
        "--config-preset",
        choices=tuple(list_presets()),
        default=settings.get("config_preset"),
        help="Optional explicit UNet preset (overrides dataset default).",
    )

    # Objective/sweep knobs.
    parser.add_argument(
        "--prediction-space",
        choices=PARAMETERIZATION_CHOICES,
        default=settings["prediction_space"],
    )
    parser.add_argument(
        "--loss-space",
        choices=PARAMETERIZATION_CHOICES,
        default=settings["loss_space"],
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=settings["n_train"],
        help="Number of real images used for training.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=settings["n_eval"],
        help="Number of real held-out/reference images used for evaluation.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=settings["n_steps"],
        help="Number of optimizer steps.",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=settings["ckpt_every"],
        help="Save/evaluate a checkpoint every this many optimizer steps.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=settings["log_every"],
        help="Print/log training loss every this many optimizer steps.",
    )
    parser.add_argument(
        "--batch-cap",
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
        "--compare-numerics",
        action="store_true",
        default=settings["compare_numerics"],
        help="Benchmark fp32 against the selected precision without writing a run.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=settings["eval_samples"],
        help="Number of generated samples per checkpoint; also z_init.pt size.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=settings["sampling_steps"],
        help="Solver steps used to generate images from z_init.",
    )

    # Data location.
    parser.add_argument("--root", default=settings["root"])
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=settings["download"],
    )

    # W&B basics.
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=settings["wandb"],
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online"),
        default=settings["wandb_mode"],
    )
    parser.add_argument("--wandb-project", default=settings["wandb_project"])
    parser.add_argument("--wandb-name", default=settings["wandb_name"])
    parser.add_argument("--wandb-group", default=settings["wandb_group"])
    parser.add_argument("--wandb-tags", default=settings["wandb_tags"])

    # Fixed-noise generation trace.
    parser.add_argument(
        "--generation-trace",
        action=argparse.BooleanOptionalAction,
        default=settings["generation_trace"],
        help="Save/display train and generated-sample image grids and log them to W&B.",
    )
    parser.add_argument(
        "--generation-trace-count",
        type=int,
        default=settings["generation_trace_count"],
        help="Number of images per row in generation trace grids.",
    )
    parser.add_argument(
        "--generation-trace-show",
        action=argparse.BooleanOptionalAction,
        default=settings["generation_trace_show"],
        help="Display generation trace figures inline when running from a notebook.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entrypoint for the image UNet recipe."""
    args = parse_args(argv)
    settings = vars(args)
    config = build_config(settings)

    if args.log_every is not None and args.log_every < 1:
        raise ValueError(f"log-every must be >= 1, got {args.log_every}")
    if args.generation_trace_count < 1:
        raise ValueError(
            f"generation-trace-count must be >= 1, got {args.generation_trace_count}"
        )
    if args.compare_numerics:
        return compare_numerics(config, args.device)

    result = run_full_job(config, args.out_dir, args.device, evaluate=True)
    _print_result(result)
    return result["run_dir"]


def _print_kv(label, value):
    print(f"{label:<18} {value}")


def _print_result(result):
    # Keep this summary easy to scan in a running terminal.
    _print_model_summary(
        result["config"],
        model_param_summary=result.get("model_param_summary"),
    )
    _print_kv("run_dir:", result["run_dir"])
    _print_kv("checkpoints:", result["artifacts"]["checkpoints"]["path"])
    _print_kv("figures:", result["artifacts"]["figures"]["path"])
    _print_kv("metrics:", result["artifacts"]["checkpoint_metrics"]["path"])
    _print_kv("loss_eval:", result["artifacts"]["loss_eval"]["path"])
    _print_kv(
        "sample_previews:",
        result["artifacts"]["checkpoint_sample_previews"]["path"],
    )


def _print_numerics_compare(result):
    rows = result["results"]
    print("=== NUMERICS_COMPARE ===")
    print(
        f"device={result['device']} n_train={result['n_train']} "
        f"batch_cap={result['batch_cap']} n_steps={result['n_steps']}"
    )
    print(
        f"{'precision':<10} {'avg_step_s':>11} {'peak_gb':>9} "
        f"{'first_loss':>12} {'final_loss':>12} {'tf32':>6} {'autocast':>10}"
    )
    for row in rows:
        first_loss = row["first_loss"]
        final_loss = row["final_loss"]
        print(
            f"{row['precision']:<10} "
            f"{row['avg_step_time_sec']:>11.4f} "
            f"{row['peak_cuda_memory_gb']:>9.3f} "
            f"{first_loss if first_loss is not None else float('nan'):>12.5g} "
            f"{final_loss if final_loss is not None else float('nan'):>12.5g} "
            f"{str(row['tf32_enabled']):>6} "
            f"{row['autocast_dtype']:>10}"
        )


def _summarize_bundle(bundle):
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
    if bundle.get("train_labels") is not None:
        summary["train_labels_shape"] = list(bundle["train_labels"].shape)
        summary["train_labels_preview"] = _preview_tensor(bundle["train_labels"])
    if bundle.get("reference_labels") is not None:
        summary["reference_labels_shape"] = list(bundle["reference_labels"].shape)
        summary["reference_labels_preview"] = _preview_tensor(bundle["reference_labels"])
    return summary


def _preview_tensor(tensor, limit=10):
    if tensor is None:
        return None
    return [int(item) for item in tensor.flatten()[:limit].tolist()]


if __name__ == "__main__":
    main()
