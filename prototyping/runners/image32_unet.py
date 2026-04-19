"""
Readable prototype recipe for 32x32 image datasets with the UNet adapter.
"""

import argparse
from pathlib import Path
import sys

# Direct script execution from prototyping/runners should still find repo packages.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototyping.runners.common import (
    PARAMETERIZATION_CHOICES,
    artifact_paths,
    as_int_tuple,
    count_parameters,
)
from prototyping.runners.config_presets import list_presets
from prototyping.runners.metrics_image32 import (
    collect_loss_eval_record,
    collect_reference_memory,
    evaluate_image32_run,
)
from prototyping.runners.image32_unet_config_builder import (
    build_config as _build_config,
    get_default_settings as _default_settings,
    list_datasets as _list_datasets,
)

import torch
from gdfm import GDFM
from mp_config import config_to_dict
from mp_data import build_data_bundle
from mp_training import train_run


def default_settings():
    """Return the canonical Image32 default settings for this recipe."""
    return _default_settings()


def build_config(settings):
    """Build the canonical config for one Image32 UNet run."""
    return _build_config(settings)


def build_data(config):
    """Materialize the canonical Image32 train/reference bundle."""
    return build_data_bundle(config)


def build_model(config, device):
    """Build the canonical GDFM wrapper and its UNet backbone."""
    return GDFM(config).to(device)


def run_one_update(model, train_tensor, config, device):
    """Run one visible optimizer update against the first mini-batch."""
    model.train()
    # This diagnostic is deterministic; full training uses canonical sampling.
    batch_size = min(int(config.training.batch_cap), int(train_tensor.shape[0]))
    batch = train_tensor[:batch_size].to(device)
    return run_sanity_updates(model, train_tensor, config, device, n_updates=1)[-1]


def run_sanity_updates(model, train_tensor, config, device, n_updates=1):
    """Run a small number of optimizer steps for command-line sanity checks."""
    if n_updates is None:
        n_updates = 1
    if int(n_updates) < 1:
        raise ValueError(f"n_updates must be >= 1, got {n_updates}")

    model.train()
    batch_size = min(int(config.training.batch_cap), int(train_tensor.shape[0]))
    if batch_size < 1:
        raise ValueError("Sanity updates require at least one training sample.")

    tensor = train_tensor.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
    )
    results = []
    base_seed = int(config.training.seed)
    for step in range(int(n_updates)):
        torch.manual_seed(base_seed + step)
        batch = tensor[:batch_size]
        eps = torch.randn_like(batch)
        t = torch.linspace(
            float(config.schedule.t_min),
            float(config.schedule.t_max),
            batch_size,
            device=batch.device,
        )

        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch, eps=eps, t=t)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0e9)
        optimizer.step()

        results.append(
            {
                "step": step + 1,
                "batch_shape": list(batch.shape),
                "loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "optimizer": "adam",
                "prediction_space": str(config.framework.prediction_space),
                "loss_space": str(config.framework.loss_space),
            }
        )
    return results


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
            train,
            config,
            device,
            fixed_times=tuple(config.evaluation.loss_eval_times),
        )
    return result


def _coerce_mnist_class_balance(settings):
    """
    Avoid hard failures when MNIST class-balanced sampling gets non-multiple counts.

    MNIST loaders require class-balanced counts to be divisible by 10.
    If a user asks for non-divisible counts with balancing enabled, keep the
    requested size but disable balancing to preserve the run intent and avoid
    dataset loader crashes.
    """
    if str(settings.get("dataset")).lower() != "mnist":
        return

    fixes = []
    if bool(settings.get("balance_train_classes", False)) and int(settings.get("n_train", 0)) % 10 != 0:
        settings["balance_train_classes"] = False
        fixes.append(
            f"MNIST train subset balancing disabled: n-train={settings['n_train']} is not divisible by 10."
        )
    if bool(settings.get("balance_reference_classes", False)) and int(settings.get("n_eval", 0)) % 10 != 0:
        settings["balance_reference_classes"] = False
        fixes.append(
            f"MNIST reference balancing disabled: n-eval={settings['n_eval']} is not divisible by 10."
        )
    if not fixes:
        return

    print("run setup:")
    for fix in fixes:
        print("  - {}".format(fix))


def run_full_job(config, out_dir, device, evaluate=True):
    """Run canonical training, then Image32 checkpoint metrics if requested."""
    # Training remains canonical; empirical metrics live beside this runner.
    run_dir = train_run(config, Path(out_dir), device=device)
    evaluation = (
        evaluate_image32_run(
            run_dir,
            device=device,
            fixed_times=tuple(config.evaluation.loss_eval_times),
        )
        if evaluate
        else {"checkpoint_metrics": [], "loss_eval": []}
    )
    return {
        "run_dir": str(run_dir),
        "checkpoint_metrics": evaluation["checkpoint_metrics"],
        "loss_eval": evaluation["loss_eval"],
        "config": config,
        "artifacts": artifact_paths(run_dir),
    }


def _print_model_summary(config, model=None, model_param_summary=None):
    """Print concise model creation + parameterization metadata."""
    network = config.network
    framework = config.framework

    model_desc = f"GDFM + {str(network.arch).upper()} ({str(network.arch)})"
    print(
        "{:<18} {}".format(
            "model creation:",
            model_desc,
        )
    )
    print(
        "{:<18} pred={} loss={}".format(
            "model parameterization:",
            str(framework.prediction_space),
            str(framework.loss_space),
        )
    )
    print(
        "{:<18} {}".format(
            "model topology:",
            "base_ch={} channel_mult={} attn={} res_blocks={} dropout={} time_emb={}".format(
                int(network.base_channels),
                tuple(network.channel_mult),
                tuple(network.attn_resolutions),
                int(network.num_res_blocks),
                float(network.dropout),
                network.time_emb_dim,
            ),
        )
    )

    if model is None and model_param_summary is None:
        return
    if model is not None:
        params = count_parameters(model)
    else:
        params = dict(model_param_summary)
    print(
        "{:<18} {:,} total | {:,} trainable".format(
            "model params:",
            params["total"],
            params["trainable"],
        )
    )


def parse_args(argv=None):
    """Parse the Image32 UNet command line."""
    settings = default_settings()
    parser = argparse.ArgumentParser(description="Image32 UNet prototype recipe.")
    parser.add_argument("--device", default=settings["device"])
    parser.add_argument("--out-dir", default=settings["out_dir"])
    parser.add_argument("--run-name", default=settings["run_name"])
    parser.add_argument("--seed", type=int, default=settings["seed"])
    parser.add_argument("--dataset", choices=tuple(_list_datasets()), default=settings["dataset"])
    parser.add_argument(
        "--config-preset",
        choices=tuple(list_presets()),
        default=settings.get("config_preset"),
        help="Optional explicit UNet preset (overrides dataset default).",
    )
    parser.add_argument("--prediction-space", choices=PARAMETERIZATION_CHOICES, default=settings["prediction_space"])
    parser.add_argument("--loss-space", choices=PARAMETERIZATION_CHOICES, default=settings["loss_space"])
    parser.add_argument("--n-train", type=int, default=settings["n_train"])
    parser.add_argument("--n-eval", type=int, default=settings["n_eval"])
    parser.add_argument("--n-steps", type=int, default=settings["n_steps"])
    parser.add_argument("--ckpt-every", type=int, default=settings["ckpt_every"])
    parser.add_argument("--batch-cap", type=int, default=settings["batch_cap"])
    parser.add_argument("--eval-samples", type=int, default=settings["eval_samples"])
    parser.add_argument("--sampling-steps", type=int, default=settings["sampling_steps"])
    parser.add_argument("--root", default=settings["root"])
    parser.add_argument("--dataset-subdir", default=settings["dataset_subdir"])
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=settings["download"])
    parser.add_argument("--train-split", default=settings["train_split"])
    parser.add_argument("--reference-split", default=settings["reference_split"])
    parser.add_argument("--allow-same-source-reference", action=argparse.BooleanOptionalAction, default=settings["allow_same_source_reference"])
    parser.add_argument("--image-size", type=int, default=settings["image_size"])
    parser.add_argument("--image-channels", type=int, choices=(1, 3), default=settings["image_channels"])
    parser.add_argument("--resize-size", type=int, default=settings["resize_size"])
    parser.add_argument("--balance-train-classes", action=argparse.BooleanOptionalAction, default=settings["balance_train_classes"])
    parser.add_argument("--balance-reference-classes", action=argparse.BooleanOptionalAction, default=settings["balance_reference_classes"])
    parser.add_argument("--loss-eval-times", default=",".join(str(item) for item in settings["loss_eval_times"]))
    parser.add_argument("--base-channels", type=int, default=settings["base_channels"])
    parser.add_argument("--channel-mult", default=",".join(str(item) for item in settings["channel_mult"]))
    parser.add_argument("--attn-resolutions", default=",".join(str(item) for item in settings["attn_resolutions"]))
    parser.add_argument("--num-res-blocks", type=int, default=settings["num_res_blocks"])
    parser.add_argument("--dropout", type=float, default=settings["dropout"])
    parser.add_argument("--time-emb-dim", type=int, default=settings["time_emb_dim"])
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=settings["wandb"])
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default=settings["wandb_mode"])
    parser.add_argument("--wandb-project", default=settings["wandb_project"])
    parser.add_argument("--wandb-entity", default=settings["wandb_entity"])
    parser.add_argument("--wandb-notes", default=settings["wandb_notes"])
    parser.add_argument("--wandb-name", default=settings["wandb_name"])
    parser.add_argument("--wandb-group", default=settings["wandb_group"])
    parser.add_argument("--wandb-tags", default=settings["wandb_tags"])
    parser.add_argument(
        "--wandb-strict",
        action=argparse.BooleanOptionalAction,
        default=settings["wandb_strict"],
    )
    parser.add_argument("--sanity-updates", type=int, default=settings["sanity_updates"])
    parser.add_argument(
        "--sanity-only", action=argparse.BooleanOptionalAction, default=settings["sanity_only"]
    )
    args = parser.parse_args(argv)
    args.channel_mult = as_int_tuple(args.channel_mult)
    args.attn_resolutions = as_int_tuple(args.attn_resolutions)
    args.loss_eval_times = _as_float_tuple(args.loss_eval_times)
    return args


def main(argv=None):
    """CLI entrypoint for the Image32 UNet recipe."""
    args = parse_args(argv)
    settings = {**default_settings(), **vars(args)}
    _coerce_mnist_class_balance(settings)
    config = build_config(settings)
    model_param_summary = None

    if args.sanity_updates is None:
        args.sanity_updates = 0
    if args.sanity_only and args.sanity_updates < 1:
        args.sanity_updates = 1
    if args.sanity_updates < 0:
        raise ValueError(f"sanity-updates must be >= 0, got {args.sanity_updates}")

    if args.sanity_updates > 0:
        data_bundle = build_data(config)
        model = build_model(config, args.device)
        model_param_summary = count_parameters(model)
        updates = run_sanity_updates(model, data_bundle["train"], config, args.device, args.sanity_updates)
        metrics = inspect_metrics(config, data_bundle, model, args.device)
        result = {
            "sanity_updates": args.sanity_updates,
            "update_records": updates,
            "metrics": metrics,
            "data": _summarize_bundle(data_bundle),
        }
        if args.sanity_only:
            _print_model_summary(config, model)
        _print_sanity_result(result)
        if args.sanity_only:
            return result
        del model

    result = run_full_job(config, args.out_dir, args.device, evaluate=True)
    if model_param_summary is not None:
        result["model_param_summary"] = model_param_summary
    _print_result(result)
    return result["run_dir"]


def _print_result(result):
    # Keep this summary easy to scan in a running terminal.
    _print_model_summary(
        result["config"],
        model_param_summary=result.get("model_param_summary"),
    )
    print("{:<18} {}".format("run_dir:", result["run_dir"]))
    print("{:<18} {}".format("checkpoints:", result["artifacts"]["checkpoints"]["path"]))
    print("{:<18} {}".format("metrics:", result["artifacts"]["checkpoint_metrics"]["path"]))
    print("{:<18} {}".format("loss_eval:", result["artifacts"]["loss_eval"]["path"]))


def _print_sanity_result(result):
    # Sanity run summaries focus on a stable small set of diagnostics.
    print("{:<18} {}".format("sanity_updates:", result["sanity_updates"]))
    print("{:<18} {}".format("train_shape:", result["data"]["train_shape"]))
    for update in result["update_records"]:
        print(
            "{:<18} step={} loss={:.6f} grad={:.6f}".format(
                "update:",
                update["step"],
                update["loss"],
                update["grad_norm"],
            )
        )


def _config_sections(config):
    payload = config_to_dict(config)
    return {
        "study": payload["study"],
        "framework": payload["framework"],
        "data": payload["data"],
        "network": payload["network"],
        "training": payload["training"],
        "evaluation": payload["evaluation"],
        "sampling": payload["sampling"],
        "outputs": payload["outputs"],
    }


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


def _as_float_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    if value is None or str(value).strip() == "":
        return ()
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


if __name__ == "__main__":
    main()
