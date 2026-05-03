"""
Canonical training loop for first-party studies.

Read this file in three layers:

1. Small helpers above `train_run(...)`
   These are intentionally small: choose an optimizer, resolve DataLoader
   batching, create shared evaluation noise, and save a simple loss figure.

2. `train_run(...)`
   This is the real orchestration point. It does more than gradient descent:
   it creates the run layout, writes the exact artifacts that checkpoint
   evaluation needs, trains the model, saves checkpoints, and records run
   status.

3. The `__main__` block
   This is a tiny runnable example that builds a GMM config and writes one
   throwaway run into a temporary directory.

The most important contract in this module:
    `train_run(...)` returns an attempt directory whose files are later consumed
    by `mp_runners.evaluate_checkpoints.evaluate_run(...)`.

Study-specific sweep logic and plotting decisions should live outside this
module. This file should stay focused on the common training lifecycle.
"""

import fcntl
import io
import os
import re
import shutil
import time
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gdfm import GDFM, print_model_summary
from mp_artifacts import (
    create_run_layout,
    list_checkpoint_paths,
    load_attempt_status,
    load_json,
    mark_attempt_done,
    make_config_hash,
    save_checkpoint,
    save_config_snapshot,
    save_json,
    save_tensor,
    update_attempt_status,
)
from mp_data import build_data_bundle
from mp_metrics.metrics_images import (
    compute_projected_image_features,
    compute_projected_image_memorization_metrics,
    make_random_projection_matrix,
)
from mp_runtime import autocast_context, configure_torch_runtime
from mp_training.checkpoint_schedule import (
    checkpoint_schedule_metadata,
    resolve_checkpoint_steps,
)
from mp_tracking import AttemptTracker, resolve_wandb_identity
from mp_visualization import (
    is_image_batch,
    maybe_display_figure,
    model_context_title,
    prepare_matplotlib_cache,
    save_image_rows_figure,
)


def _build_optimizer(config, model):
    """
    Instantiate the optimizer requested by the study config.

    This is the only place in the trainer where `config.training.optimizer` is
    interpreted. Keeping the branching here makes the main training loop easier
    to read: the loop only needs an object with `zero_grad`, `step`, and a state
    dict for checkpoints.
    """
    name = str(config.training.optimizer).lower()

    if name == "adam":
        # Adam is the default for the current study configs.
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )

    if name == "sgd":
        # SGD is kept as a simple alternative for ablations or debugging.
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
            momentum=float(config.training.momentum),
        )

    raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def _resolve_batch_cap(config):
    """
    Resolve the public batch cap while preserving the old `batch_size` alias.
    """
    training = config.training
    batch_cap = getattr(training, "batch_cap", None)
    if batch_cap is None:
        batch_cap = getattr(training, "batch_size", None)
    if batch_cap is None:
        raise ValueError("Training config must define `batch_cap` or `batch_size`.")

    batch_cap = int(batch_cap)
    if batch_cap < 1:
        raise ValueError(f"training.batch_cap must be >= 1, got {batch_cap}")
    return batch_cap


def _batching_metadata(config, train_tensor):
    """
    Compute the DataLoader batching policy for one fixed saved training tensor.

    Optimizer steps are the official training x-axis. This metadata only
    records how a selected `N` maps to batches under the batch-cap policy.
    """
    n_train = int(train_tensor.shape[0])
    if n_train < 1:
        raise ValueError("Training requires at least one selected training sample.")

    batch_cap = _resolve_batch_cap(config)
    effective_batch_size = min(batch_cap, n_train)
    drop_last = n_train > batch_cap
    if drop_last:
        steps_per_epoch = n_train // effective_batch_size
    else:
        steps_per_epoch = 1
    examples_per_epoch = steps_per_epoch * effective_batch_size
    dropped_per_epoch = n_train - examples_per_epoch

    return {
        "batch_cap": int(batch_cap),
        "effective_batch_size": int(effective_batch_size),
        "drop_last": bool(drop_last),
        "steps_per_epoch": int(steps_per_epoch),
        "examples_per_epoch": int(examples_per_epoch),
        "dropped_per_epoch": int(dropped_per_epoch),
        "n_train": int(n_train),
        "n_training_steps": int(config.training.n_training_steps),
    }


def _count_parameters(model):
    """Count total and trainable model parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}


def _model_summary(config, model):
    """
    Return the model summary printed right after model construction.

    This is intentionally tied to the actual constructed object, not only the
    config, so the printed backbone class and parameter count match the run.
    """
    net = getattr(model, "net", None)
    network = config.network
    framework = config.framework
    params = _count_parameters(model)
    summary = {
        "model": "GDFM",
        "backbone": type(net).__name__ if net is not None else None,
        "network_arch": str(network.arch),
        "prediction_space": str(framework.prediction_space),
        "loss_space": str(framework.loss_space),
        "total_parameters": params["total"],
        "trainable_parameters": params["trainable"],
    }
    if str(network.arch) == "mlp":
        summary.update(
            {
                "hidden_dim": int(network.hidden_dim),
                "n_layers": int(network.n_layers),
                "time_emb_dim": int(network.time_emb_dim),
            }
        )
    else:
        summary.update(
            {
                "base_channels": int(network.base_channels),
                "channel_mult": list(network.channel_mult),
                "attn_resolutions": list(network.attn_resolutions),
                "num_res_blocks": int(network.num_res_blocks),
                "dropout": float(network.dropout),
                "time_emb_dim": getattr(network, "time_emb_dim", None),
            }
        )
    return summary


def _seed_training_rngs(config, device):
    """
    Seed the global torch RNGs that affect model init, z_init, and training loss.

    This intentionally does not enable strict deterministic algorithms: the goal
    is same-seed reproducibility under the normal kernels chosen by the run.
    """
    seed = int(config.training.seed)
    torch.manual_seed(seed)

    torch_device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
    cuda_seeded = bool(torch_device.type == "cuda" and torch.cuda.is_available())
    if cuda_seeded:
        torch.cuda.manual_seed_all(seed)

    return {
        "training_seed": seed,
        "torch_manual_seed": True,
        "cuda_manual_seed_all": cuda_seeded,
        "z_init_seed": seed + 2003,
        "dataloader_seed": seed,
        "deterministic_algorithms": False,
    }


def _log_model_summary(tracker, model, summary):
    """Print and persist a compact or full model summary."""
    console_settings = getattr(
        getattr(tracker, "config", None),
        "tracking",
        None,
    )
    compact = True
    if console_settings is not None:
        compact = bool(
            getattr(getattr(console_settings, "console", None), "compact_model_summary", True)
        )

    if compact:
        model_name = str(summary.get("backbone", "gdfm"))
        network_arch = str(summary.get("network_arch", "unknown"))
        message = (
            "model summary\n"
            f"backbone={model_name} arch={network_arch} "
            f"pred={summary['prediction_space']} loss={summary['loss_space']} "
            f"params={summary['trainable_parameters']:,}/{summary['total_parameters']:,}"
        )
        tracker.note(
            message,
            event_type="model_summary",
            model_summary=summary,
        )
        return

    stream = io.StringIO()
    with redirect_stdout(stream):
        print_model_summary(model)
    message = "model summary\n" + stream.getvalue().rstrip()
    tracker.note(
        message,
        event_type="model_summary",
        model_summary=summary,
    )


def _data_summary(config, data_bundle, train_tensor, reference_tensor, batching):
    """Return the data summary printed after bundle construction."""
    summary = {
        "dataset_key": str(getattr(config.study, "dataset_key", "")),
        "dataset": str(config.data.dataset),
        "root": str(config.data.root),
        "train_shape": list(train_tensor.shape),
        "reference_shape": list(reference_tensor.shape),
        "display_shape": data_bundle.get("display_shape"),
        "train_split": data_bundle.get("train_split"),
        "reference_split": data_bundle.get("reference_split")
        or getattr(config.data, "reference_split", None),
        "train_role": data_bundle.get("train_role"),
        "reference_role": data_bundle.get("reference_role"),
        "train_source_id": data_bundle.get("train_source_id"),
        "reference_source_id": data_bundle.get("reference_source_id"),
        "train_class_counts": data_bundle.get("train_class_counts"),
        "reference_class_counts": data_bundle.get("reference_class_counts"),
        "train_labels_shape": _tensor_shape(data_bundle.get("train_labels")),
        "reference_labels_shape": _tensor_shape(data_bundle.get("reference_labels")),
        "data_validation": data_bundle.get("data_validation"),
        "batching": batching,
    }
    geometry = data_bundle.get("geometry")
    if isinstance(geometry, dict):
        summary["geometry_keys"] = sorted(str(key) for key in geometry.keys())
    return summary


def _log_data_summary(tracker, summary):
    """Print and persist the data summary in a readable block."""
    validation = summary.get("data_validation") or {}
    lines = [
        "data summary",
        _format_readout_row(
            "dataset",
            (
                f"key={summary['dataset_key']} internal={summary['dataset']} "
                f"root={summary['root']}"
            ),
        ),
        _format_readout_row("display_shape", summary["display_shape"]),
        _format_readout_row(
            "data train",
            _format_fields(
                [
                    ("shape", summary["train_shape"]),
                    ("split", summary["train_split"]),
                    ("source", summary["train_source_id"]),
                    ("role", summary["train_role"]),
                ]
            ),
        ),
        _format_readout_row(
            "data reference",
            _format_fields(
                [
                    ("shape", summary["reference_shape"]),
                    ("split", summary["reference_split"]),
                    ("source", summary["reference_source_id"]),
                    ("role", summary["reference_role"]),
                ]
            ),
        ),
        _format_readout_row(
            "data leakage_check",
            _format_fields(
                [
                    ("same_source", validation.get("same_source")),
                    ("overlap_checked", validation.get("overlap_checked")),
                    ("overlap_count", validation.get("overlap_count")),
                    ("same_source_allowed", validation.get("same_source_allowed")),
                ]
            ),
        ),
        _format_readout_row(
            "data batching",
            _format_fields(
                [
                    ("batch_cap", summary["batching"]["batch_cap"]),
                    ("effective", summary["batching"]["effective_batch_size"]),
                    ("drop_last", summary["batching"]["drop_last"]),
                    ("steps_per_epoch", summary["batching"]["steps_per_epoch"]),
                ]
            ),
        ),
    ]
    if summary.get("train_class_counts") is not None:
        lines.append(
            _format_readout_row(
                "data class_counts",
                (
                    f"train={summary['train_class_counts']} "
                    f"reference={summary['reference_class_counts']}"
                ),
            )
        )
    if summary.get("geometry_keys") is not None:
        lines.append(_format_readout_row("data geometry_keys", summary["geometry_keys"]))
    _emit_readout_block(
        tracker,
        "DATA BUILDER SUMMARY",
        lines,
        event_type="data_summary",
        event_fields={"data_summary": summary},
    )


def _format_readout_row(label, value):
    """Return one aligned terminal-summary row."""
    return f"  {str(label):<18} {value}"


def _format_fields(pairs):
    """Format short key/value fields on one line."""
    return " ".join(f"{key}={value}" for key, value in pairs)


def _emit_readout_block(tracker, title, lines, *, event_type, event_fields):
    """Emit a compact multi-line readout to terminal, log, and events."""
    width = max(72, len(str(title)) + 8)
    border = "-" * width
    block_lines = [border, f"{str(title).center(width)}", border, *lines, border]
    message = "\n".join(block_lines)
    tracker.note(
        message,
        event_type=event_type,
        **event_fields,
    )


def _tensor_shape(value):
    """Return a tensor shape as a JSON-friendly list."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return list(value.shape)
    return None


def _build_train_loader(train_tensor, batching, generator, *, pin_memory=False):
    """
    Build a shuffled DataLoader over the already-materialized training tensor.
    """
    return DataLoader(
        train_tensor,
        batch_size=int(batching["effective_batch_size"]),
        shuffle=True,
        drop_last=bool(batching["drop_last"]),
        generator=generator,
        pin_memory=bool(pin_memory),
    )


def _configured_stop_after_step(config):
    """
    Return the effective final step for this process.

    `training.n_training_steps` remains the logical run target. An optional
    `stop_after_step` lets Slurm chunks stop early while preserving absolute
    optimizer-step numbering and W&B axes.
    """
    logical_target = int(config.training.n_training_steps)
    if logical_target < 1:
        raise ValueError(
            f"training.n_training_steps must be >= 1, got {logical_target}"
        )
    raw_stop = getattr(config.training, "stop_after_step", None)
    if raw_stop in (None, ""):
        return logical_target
    stop_after = int(raw_stop)
    if stop_after < 1:
        raise ValueError(f"training.stop_after_step must be >= 1, got {stop_after}")
    if stop_after > logical_target:
        raise ValueError(
            "training.stop_after_step cannot exceed training.n_training_steps: "
            f"{stop_after} > {logical_target}"
        )
    return stop_after


def _resume_path_value(config, field):
    return str(getattr(config.training, field, "") or "").strip()


def _checkpoint_step_from_path(path):
    match = re.match(r"^step_(\d+)\.pt$", Path(path).name)
    if not match:
        raise ValueError(f"Checkpoint path does not look like step_XXXXXX.pt: {path}")
    return int(match.group(1))


def _resolve_resume_source(config):
    """
    Resolve resume CLI/config fields into one source attempt and checkpoint.
    """
    resume_attempt = _resume_path_value(config, "resume_from_attempt")
    resume_checkpoint = _resume_path_value(config, "resume_from_checkpoint")
    if resume_attempt and resume_checkpoint:
        raise ValueError(
            "training.resume_from_attempt and training.resume_from_checkpoint "
            "are mutually exclusive."
        )
    if not resume_attempt and not resume_checkpoint:
        return None

    if resume_attempt:
        source_attempt_dir = Path(resume_attempt).expanduser().resolve()
        checkpoints = list_checkpoint_paths(source_attempt_dir)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found under {source_attempt_dir / 'checkpoints'}"
            )
        checkpoint_path = Path(checkpoints[-1]).resolve()
    else:
        checkpoint_path = Path(resume_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        if checkpoint_path.parent.name != "checkpoints":
            raise ValueError(
                "Resume checkpoint must live under an attempt checkpoints/ directory: "
                f"{checkpoint_path}"
            )
        source_attempt_dir = checkpoint_path.parent.parent.resolve()

    status = load_attempt_status(source_attempt_dir)
    source_step = _checkpoint_step_from_path(checkpoint_path)
    return {
        "source_attempt_dir": source_attempt_dir,
        "source_checkpoint": checkpoint_path,
        "source_step": int(source_step),
        "source_status": status,
    }


def _validate_resume_source(config, resume_source, target_step):
    """
    Reject resume attempts that would combine incompatible semantic runs.
    """
    if resume_source is None:
        return
    source_status = resume_source["source_status"]
    source_hash = source_status.get("config_hash")
    current_hash = make_config_hash(config)
    if source_hash and source_hash != current_hash:
        raise ValueError(
            "Resume source config does not match the current run config: "
            f"source={source_hash} current={current_hash}"
        )
    source_step = int(resume_source["source_step"])
    if source_step >= int(target_step):
        raise ValueError(
            "Resume checkpoint is already at or beyond the requested target: "
            f"source_step={source_step} target_step={int(target_step)}"
        )


def _resume_metadata(config, resume_source, target_step):
    if resume_source is None:
        return None
    source_status = resume_source["source_status"]
    source_wandb = source_status.get("wandb") or {}
    return {
        "source_attempt_dir": str(resume_source["source_attempt_dir"]),
        "source_checkpoint": str(resume_source["source_checkpoint"]),
        "source_step": int(resume_source["source_step"]),
        "logical_target_step": int(config.training.n_training_steps),
        "stop_after_step": (
            None
            if getattr(config.training, "stop_after_step", None) in (None, "")
            else int(config.training.stop_after_step)
        ),
        "target_step": int(target_step),
        "source_wandb_run_id": source_wandb.get("run_id"),
    }


def _set_wandb_resume_id(config, resume_source):
    if resume_source is None:
        return
    source_wandb = resume_source["source_status"].get("wandb") or {}
    run_id = str(source_wandb.get("run_id") or "").strip()
    if run_id:
        config.tracking.wandb.resume_run_id = run_id


def _acquire_resume_lock(run_dir):
    """
    Hold one advisory lock per semantic run/seed while a resume is active.
    """
    seed_dir = Path(run_dir).parents[1]
    lock_path = seed_dir / ".resume.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            "Another resume job appears to be active for this run/seed. "
            f"Lock path: {lock_path}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} attempt_dir={Path(run_dir).resolve()}\n")
    handle.flush()
    return handle


def _release_resume_lock(handle):
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _link_or_copy_file(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _optional_torch_load(path):
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _load_resume_artifacts(run_dir, resume_source):
    """
    Reuse the exact saved data/evaluation tensors from the source attempt.
    """
    source_artifacts = Path(resume_source["source_attempt_dir"]) / "artifacts"
    target_artifacts = Path(run_dir) / "artifacts"
    required = ("train_tensor.pt", "reference_tensor.pt", "z_init.pt")
    for name in required:
        source = source_artifacts / name
        if not source.exists():
            raise FileNotFoundError(f"Resume source is missing required artifact: {source}")
        _link_or_copy_file(source, target_artifacts / name)

    optional = (
        "train_labels.pt",
        "reference_labels.pt",
        "train_indices.pt",
        "reference_indices.pt",
        "geometry.pt",
    )
    for name in optional:
        source = source_artifacts / name
        if source.exists():
            _link_or_copy_file(source, target_artifacts / name)

    train_tensor = torch.load(target_artifacts / "train_tensor.pt", map_location="cpu")
    reference_tensor = torch.load(
        target_artifacts / "reference_tensor.pt",
        map_location="cpu",
    )
    z_init = torch.load(target_artifacts / "z_init.pt", map_location="cpu")

    data_meta_path = source_artifacts / "data_meta.json"
    data_meta = load_json(data_meta_path) if data_meta_path.exists() else {}
    data_bundle = {
        "train": train_tensor,
        "reference": reference_tensor,
        "display_shape": data_meta.get("display_shape"),
        "train_split": data_meta.get("train_split"),
        "reference_split": data_meta.get("reference_split"),
        "train_role": data_meta.get("train_role"),
        "reference_role": data_meta.get("reference_role"),
        "train_source_id": data_meta.get("train_source_id"),
        "reference_source_id": data_meta.get("reference_source_id"),
        "train_class_counts": data_meta.get("train_class_counts"),
        "reference_class_counts": data_meta.get("reference_class_counts"),
        "requested_balanced_train_classes": data_meta.get(
            "requested_balanced_train_classes"
        ),
        "requested_balanced_reference_classes": data_meta.get(
            "requested_balanced_reference_classes"
        ),
        "balanced_train_classes": data_meta.get("balanced_train_classes"),
        "balanced_reference_classes": data_meta.get("balanced_reference_classes"),
        "class_balance_adjustments": data_meta.get("class_balance_adjustments"),
        "data_validation": data_meta.get("data_validation"),
        "train_labels": _optional_torch_load(target_artifacts / "train_labels.pt"),
        "reference_labels": _optional_torch_load(
            target_artifacts / "reference_labels.pt"
        ),
        "train_indices": _optional_torch_load(target_artifacts / "train_indices.pt"),
        "reference_indices": _optional_torch_load(
            target_artifacts / "reference_indices.pt"
        ),
        "geometry": _optional_torch_load(target_artifacts / "geometry.pt"),
    }
    return data_bundle, train_tensor, reference_tensor, z_init


def _link_resume_checkpoint(run_dir, resume_source):
    source = Path(resume_source["source_checkpoint"])
    destination = Path(run_dir) / "checkpoints" / source.name
    return _link_or_copy_file(source, destination)


def _load_resume_history(resume_source):
    if resume_source is None:
        return []
    history_path = Path(resume_source["source_attempt_dir"]) / "metrics" / "train_history.json"
    if not history_path.exists():
        return []
    try:
        history = load_json(history_path)
    except Exception:
        return []
    source_step = int(resume_source["source_step"])
    filtered = []
    for record in history if isinstance(history, list) else []:
        try:
            step = int(record.get("step"))
        except (TypeError, ValueError):
            continue
        if step <= source_step:
            filtered.append(record)
    return filtered


def _make_z_init(config, train_tensor):
    """
    Create the shared initial noise tensor used for later checkpoint evaluation.

    Saving this once per run is important for fair comparisons across
    checkpoints and parameterizations: evaluation should start from the same
    initial noise, not a freshly sampled one every time.

    The shape comes from the training tensor after data loading. That keeps this
    helper agnostic to vectors vs images: GMM gives `[B, D]`, while image data
    gives `[B, C, H, W]`.
    """
    sample_shape = tuple(train_tensor.shape[1:])
    n_samples = int(
        getattr(
            config.evaluation,
            "sample_bank_size",
            getattr(config.evaluation, "n_samples", 64),
        )
    )
    generation_monitor = getattr(config, "generation_monitor", None)
    if generation_monitor is not None and bool(
        getattr(generation_monitor, "enabled", False)
    ):
        n_samples = max(n_samples, int(getattr(generation_monitor, "n_samples", 64)))
    random_projected_f_mem = _random_projected_f_mem_config(config)
    if random_projected_f_mem is not None and is_image_batch(train_tensor):
        n_samples = max(
            n_samples,
            int(getattr(random_projected_f_mem, "n_samples", 512)),
        )
    generator = torch.Generator().manual_seed(int(config.training.seed) + 2003)
    return torch.randn(n_samples, *sample_shape, generator=generator)


def _save_training_loss_figure(history, figures_dir):
    """
    Save the standard loss-vs-step training curve for one attempt.

    The trainer writes both machine-readable loss history and this quick visual
    check. The figure is not required for evaluation, but it is useful when
    opening a run folder by hand.
    """
    if not history:
        return None

    # Import matplotlib lazily so importing the trainer does not require a
    # plotting backend until we actually have a figure to save.
    prepare_matplotlib_cache(Path(figures_dir).parent / "logs")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "training_loss_vs_step.png"
    steps = [item["step"] for item in history]
    losses = [item["loss"] for item in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, color="#1f77b4", linewidth=1.8)
    ax.set_title("Training loss vs step")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _generation_monitor_config(config):
    """
    Return the optional generation monitor config section, if enabled.
    """
    trace = getattr(config, "generation_monitor", None)
    if trace is None or not bool(getattr(trace, "enabled", False)):
        return None
    return trace


def _random_projected_f_mem_config(config):
    """
    Return random projected f_mem config, if present.
    """
    return getattr(config, "random_projected_f_mem", None)


def _configured_monitor_batch_size(config):
    """
    Return the configured no-grad train-monitor sampling batch size.

    Older saved configs may not have a `monitoring` section, so fall back to
    evaluation batch size and then training batch cap.
    """
    monitoring = getattr(config, "monitoring", None)
    batch_size = (
        getattr(monitoring, "batch_size", None) if monitoring is not None else None
    )
    if batch_size is None:
        evaluation = getattr(config, "evaluation", None)
        batch_size = (
            getattr(evaluation, "batch_size", None) if evaluation is not None else None
        )
    if batch_size is None:
        training = getattr(config, "training", None)
        batch_size = (
            getattr(training, "batch_cap", None) if training is not None else None
        )
    if batch_size is None:
        batch_size = 1
    return max(1, int(batch_size))


def _monitor_batch_size(config, count):
    """Return the effective no-grad train-monitor sampling batch size."""
    return max(1, min(_configured_monitor_batch_size(config), int(count)))


def _format_monitor_seconds(seconds):
    """Return a compact seconds string for monitor timing notes."""
    return f"{float(seconds):.3f}s"


def _prepare_random_projected_f_mem(config, train_tensor, run_dir, tracker, device):
    """
    Save the fixed projection used by the checkpoint-time monitor.
    """
    monitor = _random_projected_f_mem_config(config)
    if monitor is None:
        return {"enabled": False, "records": []}
    if not is_image_batch(train_tensor):
        return {"enabled": False, "skipped": "non_image_train_tensor", "records": []}

    resize_size = int(getattr(monitor, "resize_size", 16))
    projection_dim = int(getattr(monitor, "projection_dim", 256))
    projection_seed = getattr(monitor, "projection_seed", None)
    if projection_seed is None:
        projection_seed = int(config.training.seed) + 3001
    projection_seed = int(projection_seed)
    input_dim = int(train_tensor.shape[1]) * resize_size * resize_size
    projection_matrix = make_random_projection_matrix(
        input_dim,
        projection_dim,
        projection_seed,
    )
    matrix_path = Path(run_dir) / "artifacts" / "random_projected_f_mem_projection.pt"
    reference_vectors_path = (
        Path(run_dir) / "artifacts" / "random_projected_f_mem_reference_vectors.pt"
    )
    metadata_path = Path(run_dir) / "artifacts" / "random_projected_f_mem_projection.json"
    reference_projection_start = time.perf_counter()
    projected_reference_vectors = compute_projected_image_features(
        train_tensor,
        resize_size=resize_size,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
        projection_matrix=projection_matrix,
        image_range="-1_1",
        device=device,
        batch_size=_configured_monitor_batch_size(config),
    )
    reference_projection_elapsed_sec = time.perf_counter() - reference_projection_start
    metadata = {
        "metric": "projected_image_f_mem",
        "space": "random_projected_pixels",
        "n_samples": int(getattr(monitor, "n_samples", 512)),
        "monitor_batch_size": int(_configured_monitor_batch_size(config)),
        "resize_size": resize_size,
        "projection_dim": projection_dim,
        "projection_seed": projection_seed,
        "input_dim": input_dim,
        "projected_reference_cached": True,
        "reference_vectors_shape": list(projected_reference_vectors.shape),
        "reference_projection_elapsed_sec": reference_projection_elapsed_sec,
        "projection_matrix_path": "artifacts/random_projected_f_mem_projection.pt",
        "reference_vectors_path": (
            "artifacts/random_projected_f_mem_reference_vectors.pt"
        ),
        "metadata_path": "artifacts/random_projected_f_mem_projection.json",
    }
    save_tensor(matrix_path, projection_matrix)
    save_tensor(reference_vectors_path, projected_reference_vectors.detach().cpu())
    save_json(metadata_path, metadata)
    tracker.note(
        "random projected f_mem monitor prepared "
        f"samples={metadata['n_samples']} resize={resize_size} "
        f"projection_dim={projection_dim} "
        f"reference_projection={_format_monitor_seconds(reference_projection_elapsed_sec)}",
        event_type="random_projected_f_mem_prepared",
        random_projected_f_mem=metadata,
    )
    return {
        "enabled": True,
        "projection_matrix": projection_matrix.to(
            torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        ),
        "projected_reference_vectors": projected_reference_vectors,
        "metadata": metadata,
        "records": [],
    }


def _sample_monitor_z_init(model, z_init, count, config, device):
    """
    Sample a checkpoint monitor batch from the fixed saved source noise.
    """
    count = min(int(count), int(z_init.shape[0]))
    batch_size = _monitor_batch_size(config, count)
    samples = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, count, batch_size):
                z_batch = z_init[start : start + batch_size].clone().to(device)
                with autocast_context(config, device):
                    samples.append(
                        model.sample(
                            z_batch,
                            n_steps=int(config.sampling.n_steps),
                            method=str(config.sampling.method),
                            device=device,
                        ).cpu()
                    )
    finally:
        if was_training:
            model.train()
    return torch.cat(samples, dim=0)


def _run_random_projected_f_mem(
    config,
    model,
    train_tensor,
    z_init,
    monitor_state,
    device,
    tracker,
    step,
):
    """
    Compute and log checkpoint-time random projected f_mem monitor metrics.
    """
    if not bool(monitor_state.get("enabled", False)):
        return None
    metadata = dict(monitor_state["metadata"])
    step = int(step)
    total_start = time.perf_counter()
    sampling_start = time.perf_counter()
    samples = _sample_monitor_z_init(
        model,
        z_init,
        int(metadata["n_samples"]),
        config,
        device,
    )
    sampling_elapsed_sec = time.perf_counter() - sampling_start
    metric_start = time.perf_counter()
    metrics = compute_projected_image_memorization_metrics(
        samples,
        reference=None,
        projected_reference_vectors=monitor_state["projected_reference_vectors"],
        resize_size=int(metadata["resize_size"]),
        projection_dim=int(metadata["projection_dim"]),
        projection_seed=int(metadata["projection_seed"]),
        projection_matrix=monitor_state["projection_matrix"],
        rho=float(getattr(config.evaluation, "ratio_threshold", 0.5)),
        query_batch_size=min(256, int(samples.shape[0])),
        reference_batch_size=int(config.evaluation.chunk_size),
        image_range="-1_1",
        device=device,
    )
    metric_elapsed_sec = time.perf_counter() - metric_start
    total_elapsed_sec = time.perf_counter() - total_start
    record = {
        "step": step,
        "sampling_elapsed_sec": sampling_elapsed_sec,
        "metric_elapsed_sec": metric_elapsed_sec,
        "total_elapsed_sec": total_elapsed_sec,
        **metrics,
    }
    monitor_state["records"].append(record)
    tracker.note(
        "random projected f_mem monitor "
        f"step={step} samples={int(samples.shape[0])} "
        f"sampling={_format_monitor_seconds(sampling_elapsed_sec)} "
        f"metric={_format_monitor_seconds(metric_elapsed_sec)} "
        f"total={_format_monitor_seconds(total_elapsed_sec)}",
        event_type="random_projected_f_mem_timing",
        step=step,
        n_samples=int(samples.shape[0]),
        sampling_elapsed_sec=sampling_elapsed_sec,
        metric_elapsed_sec=metric_elapsed_sec,
        total_elapsed_sec=total_elapsed_sec,
    )
    tracker.log_metrics(
        {
            "monitor/random_projected_f_mem": record["f_mem"],
            "monitor/random_projected_f_mem_mean_r": record["mean_r"],
            "monitor/random_projected_f_mem_std_r": record["std_r"],
            "monitor/random_projected_f_mem_sampling_seconds": sampling_elapsed_sec,
            "monitor/random_projected_f_mem_metric_seconds": metric_elapsed_sec,
            "monitor/random_projected_f_mem_total_seconds": total_elapsed_sec,
        },
        step=step,
    )
    return record


def _generation_monitor_train_count(config, train_tensor):
    """
    Return the number of train examples used in the dataset grid.
    """
    trace = _generation_monitor_config(config)
    if trace is None:
        return 0
    return max(
        1,
        min(int(getattr(trace, "n_samples", 8)), int(train_tensor.shape[0])),
    )


def _generation_monitor_sample_count(config, z_init):
    """
    Return the number of generated samples shown in generation monitor figures.
    """
    trace = _generation_monitor_config(config)
    if trace is None:
        return 0
    return max(
        1,
        min(int(getattr(trace, "n_samples", 8)), int(z_init.shape[0])),
    )


def _fork_rng_devices_for(device):
    """
    Scope RNG preservation to the requested CUDA device when one is active.
    """
    if isinstance(device, int):
        torch_device = torch.device(f"cuda:{device}")
    else:
        torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return []
    if torch_device.index is None:
        return [torch.cuda.current_device()]
    return [int(torch_device.index)]


def _save_generation_monitor_train_grid(config, train_tensor, run_dir, tracker):
    """
    Save/log the train grid before any model is constructed.

    This records what the selected training subset looks like before generated
    samples are added to the same figure style.
    """
    trace = _generation_monitor_config(config)
    if trace is None:
        return None
    if not is_image_batch(train_tensor):
        tracker.warn(
            "generation monitor train grid skipped: expected image tensor shaped [B,C,H,W]"
        )
        return None

    count = _generation_monitor_train_count(config, train_tensor)
    figure_path = run_dir / "figures" / "generation_monitor_train_grid.png"
    start_time = time.perf_counter()
    save_image_rows_figure(
        [train_tensor[:count]],
        ["train"],
        figure_path,
        title=model_context_title(config, "Train grid before model creation"),
    )
    tracker.register_figure("generation_monitor_train_grid", figure_path)
    maybe_display_figure(figure_path, bool(getattr(trace, "show", True)))
    elapsed_sec = time.perf_counter() - start_time
    tracker.note(
        "generation monitor train grid "
        f"samples={count} total={_format_monitor_seconds(elapsed_sec)}",
        event_type="generation_monitor_timing",
        monitor_kind="train_grid",
        n_samples=count,
        total_elapsed_sec=elapsed_sec,
    )
    tracker.log_metrics(
        {"generation_monitor/train_grid_total_seconds": elapsed_sec},
        step=0,
    )
    return figure_path


def _save_generation_monitor_samples_figure(
    model,
    z_init,
    config,
    device,
    figure_path,
    *,
    title,
):
    """
    Sample from a model and save a generated-only image grid.
    """
    if not is_image_batch(z_init):
        return None

    count = _generation_monitor_sample_count(config, z_init)
    if count < 1:
        return None

    samples = _sample_monitor_z_init(model, z_init, count, config, device)

    return save_image_rows_figure(
        [samples],
        ["generated"],
        figure_path,
        title=title,
    )


def _save_final_generation_monitor_samples(
    config,
    model,
    train_tensor,
    z_init,
    run_dir,
    device,
    tracker,
):
    """
    Save/log final generated images from the fixed saved z_init bank.

    The fixed saved `z_init` is reused here so the final panel is comparable to
    checkpoint panels from the same run.
    """
    trace = _generation_monitor_config(config)
    if trace is None:
        return None
    if not is_image_batch(z_init):
        return None

    count = _generation_monitor_sample_count(config, z_init)
    start_time = time.perf_counter()
    with torch.random.fork_rng(devices=_fork_rng_devices_for(device), enabled=True):
        torch.manual_seed(int(config.training.seed) + 2003)
        figure_path = _save_generation_monitor_samples_figure(
            model,
            z_init,
            config,
            device,
            run_dir / "figures" / "generation_monitor_samples_final.png",
            title=model_context_title(config, "Final generation monitor samples"),
        )
    elapsed_sec = time.perf_counter() - start_time
    if figure_path is None:
        return None

    step = int(config.training.n_training_steps)
    tracker.register_figure(
        "generation_monitor_samples_final",
        figure_path,
        log_to_wandb=False,
    )
    tracker.log_figure_metric(
        "generation_monitor/final_samples",
        figure_path,
        step=step,
    )
    maybe_display_figure(figure_path, bool(getattr(trace, "show", True)))
    tracker.note(
        "generation monitor final samples "
        f"step={step} samples={count} total={_format_monitor_seconds(elapsed_sec)}",
        event_type="generation_monitor_timing",
        monitor_kind="final_samples",
        step=step,
        n_samples=count,
        total_elapsed_sec=elapsed_sec,
    )
    tracker.log_metrics(
        {"generation_monitor/final_samples_total_seconds": elapsed_sec},
        step=step,
    )
    return figure_path


def _save_generation_monitor_checkpoint_samples(
    config,
    model,
    train_tensor,
    z_init,
    run_dir,
    device,
    tracker,
    step,
):
    """
    Save/log generated checkpoint samples from the fixed saved z_init bank.
    """
    trace = _generation_monitor_config(config)
    if trace is None:
        return None
    if not is_image_batch(z_init):
        return None

    step = int(step)
    figure_name = f"generation_monitor_samples_step_{step:06d}"
    count = _generation_monitor_sample_count(config, z_init)
    start_time = time.perf_counter()
    figure_path = _save_generation_monitor_samples_figure(
        model,
        z_init,
        config,
        device,
        run_dir / "figures" / f"{figure_name}.png",
        title=model_context_title(
            config,
            "Generation monitor samples",
            checkpoint_step=step,
            extra="fixed_z_init",
        ),
    )
    elapsed_sec = time.perf_counter() - start_time
    if figure_path is None:
        return None

    tracker.register_figure(figure_name, figure_path, log_to_wandb=False)
    tracker.log_figure_metric(
        "generation_monitor/samples",
        figure_path,
        step=step,
    )
    maybe_display_figure(figure_path, bool(getattr(trace, "show", True)))
    tracker.note(
        "generation monitor checkpoint samples "
        f"step={step} samples={count} total={_format_monitor_seconds(elapsed_sec)}",
        event_type="generation_monitor_timing",
        monitor_kind="checkpoint_samples",
        step=step,
        n_samples=count,
        total_elapsed_sec=elapsed_sec,
    )
    tracker.log_metrics(
        {"generation_monitor/checkpoint_samples_total_seconds": elapsed_sec},
        step=step,
    )
    return {
        "step": step,
        "path": str(Path(figure_path).resolve()),
        "z_init_path": str((Path(run_dir) / "artifacts" / "z_init.pt").resolve()),
        "n_samples": count,
        "total_elapsed_sec": elapsed_sec,
    }


def train_run(config, study_dir, device="cpu"):
    """
    Run one complete training job and return the created run directory.

    Mental model:
        - `study_dir` is the experiment root chosen by the caller.
        - `create_run_layout(...)` creates one concrete attempt directory under
          that root.
        - Everything saved under that attempt directory is part of the
          evaluation contract for `evaluate_run(...)`.

    Phases:
        1. Create the run layout on disk.
        2. Start attempt tracking.
        3. Build the canonical data bundle.
        4. Save evaluation-critical artifacts: config, tensors, labels,
           indices, geometry, metadata, and shared `z_init`.
        5. Build the GDFM model and optimizer.
        6. Train, log loss, and save checkpoints.
        7. Write compact summaries for later inspection.
        8. Mark failures or interrupts in the attempt status files.
    """
    study_dir = Path(study_dir)
    run_start = time.perf_counter()
    target_step = _configured_stop_after_step(config)
    resume_source = _resolve_resume_source(config)
    _validate_resume_source(config, resume_source, target_step)
    _set_wandb_resume_id(config, resume_source)
    resume_metadata = _resume_metadata(config, resume_source, target_step)

    # ---------------------------------------------------------------------
    # Phase 1: allocate the attempt directory.
    #
    # The returned path is not just a loose output folder. It has the canonical
    # layout:
    #
    #   <study_dir>/runs/<run_id>/seeds/<seed_id>/attempts/<attempt_id>/
    #
    # Evaluation and status tooling assume this shape, so all later paths are
    # built relative to `run_dir`.
    # ---------------------------------------------------------------------
    run_dir = create_run_layout(study_dir, config)

    # `tracker` is initialized inside the `try` so the exception handlers can
    # still update status if anything fails after layout creation.
    tracker = None
    resume_lock_handle = None
    try:
        if resume_source is not None:
            resume_lock_handle = _acquire_resume_lock(run_dir)

        # Resolve numerical mode and H2D movement before the tracker starts so
        # the saved config and W&B config include the exact runtime settings.
        runtime = configure_torch_runtime(config, device)
        seed_policy = _seed_training_rngs(config, device)

        checkpoint_steps = tuple(
            step for step in resolve_checkpoint_steps(config) if int(step) <= target_step
        )
        if int(target_step) not in checkpoint_steps:
            checkpoint_steps = tuple(sorted((*checkpoint_steps, int(target_step))))
        checkpoint_step_set = set(checkpoint_steps)
        checkpoint_schedule = checkpoint_schedule_metadata(config, checkpoint_steps)
        config.training.resolved_ckpt_steps = tuple(checkpoint_steps)
        resolve_wandb_identity(config)

        # -----------------------------------------------------------------
        # Phase 2: start operator-facing tracking.
        #
        # AttemptTracker owns status.json, events.jsonl, console summaries, and
        # optional wandb forwarding. The training loop below only calls small
        # tracker methods at lifecycle boundaries.
        # -----------------------------------------------------------------
        tracker = AttemptTracker(config, run_dir)
        tracker.note(f"attempt_dir={run_dir.resolve()}", event_type="attempt_started")
        if resume_metadata is not None:
            tracker.stage(
                "resuming",
                message=(
                    "resuming "
                    f"source_step={resume_metadata['source_step']} "
                    f"target_step={resume_metadata['target_step']} "
                    f"logical_target={resume_metadata['logical_target_step']} "
                    f"checkpoint={resume_metadata['source_checkpoint']}"
                ),
                resume=resume_metadata,
            )

        # -----------------------------------------------------------------
        # Phase 3: materialize the data bundle.
        #
        # `build_data_bundle` dispatches on `config.data.dataset` and returns a
        # standard dictionary with train/reference tensors plus any labels,
        # source indices, class counts, or synthetic geometry.
        # -----------------------------------------------------------------
        if resume_source is None:
            data_bundle = build_data_bundle(config)
        else:
            data_bundle, train_tensor, reference_tensor, z_init = _load_resume_artifacts(
                run_dir,
                resume_source,
            )

        # -----------------------------------------------------------------
        # Phase 4: save the exact checkpoint-evaluation inputs.
        #
        # `evaluate_run(...)` does not rebuild the dataset from config. It loads
        # these saved tensors so checkpoint comparisons use the exact same data
        # the model trained against.
        # -----------------------------------------------------------------
        if resume_source is None:
            train_tensor = data_bundle["train"].cpu()
            reference_tensor = data_bundle["reference"].cpu()

            # `z_init` is the shared evaluation starting point. Every checkpoint is
            # sampled from this same initial noise tensor during evaluation.
            z_init = _make_z_init(config, train_tensor)

        # Resolve the public batch cap after the exact selected dataset size is
        # known. The legacy `batch_size` field mirrors the cap; derived values
        # live in `batching`.
        batching = _batching_metadata(config, train_tensor)
        config.training.batch_cap = int(batching["batch_cap"])
        config.training.batch_size = int(batching["batch_cap"])

        # Save the resolved config and the two core data tensors first. These
        # three files are the minimum needed to understand what this attempt
        # trained on.
        save_config_snapshot(run_dir, config)
        if resume_source is None:
            save_tensor(run_dir / "artifacts" / "train_tensor.pt", train_tensor)
            save_tensor(run_dir / "artifacts" / "reference_tensor.pt", reference_tensor)
            save_tensor(run_dir / "artifacts" / "z_init.pt", z_init)

            # Optional labels are present for image classes and GMM cluster IDs.
            # They are not used by the unconditional GDFM training loop, but they
            # are useful for later diagnostics and balanced-subset checks.
            if data_bundle.get("train_labels") is not None:
                save_tensor(
                    run_dir / "artifacts" / "train_labels.pt",
                    data_bundle["train_labels"],
                )
            if data_bundle.get("reference_labels") is not None:
                save_tensor(
                    run_dir / "artifacts" / "reference_labels.pt",
                    data_bundle["reference_labels"],
                )

            # Optional source indices let evaluation code and humans verify that train
            # and reference sets did not accidentally overlap when both came from a
            # shared corpus.
            if data_bundle.get("train_indices") is not None:
                save_tensor(
                    run_dir / "artifacts" / "train_indices.pt",
                    data_bundle["train_indices"],
                )
            if data_bundle.get("reference_indices") is not None:
                save_tensor(
                    run_dir / "artifacts" / "reference_indices.pt",
                    data_bundle["reference_indices"],
                )

            # Synthetic runs carry planted geometry such as a GMM basis or torus
            # radii. Saving geometry here lets metrics explain ambient-vs-intrinsic
            # behavior without reconstructing hidden state from seeds.
            if data_bundle.get("geometry") is not None:
                save_tensor(run_dir / "artifacts" / "geometry.pt", data_bundle["geometry"])

        # `data_meta.json` is the compact human-readable version of the bundle
        # contract. The heavy tensors live in `.pt` files; this JSON file tells
        # you what they are, where they came from, and which leakage checks ran.
        data_meta = {
            "dataset": str(config.data.dataset),
            "data_root": str(config.data.root),
            "download": bool(config.data.download),
            "display_shape": data_bundle.get("display_shape"),
            "train_shape": list(train_tensor.shape),
            "reference_shape": list(reference_tensor.shape),
            "train_class_counts": data_bundle.get("train_class_counts"),
            "reference_class_counts": data_bundle.get("reference_class_counts"),
            "requested_balanced_train_classes": data_bundle.get(
                "requested_balanced_train_classes"
            ),
            "requested_balanced_reference_classes": data_bundle.get(
                "requested_balanced_reference_classes"
            ),
            "balanced_train_classes": data_bundle.get("balanced_train_classes"),
            "balanced_reference_classes": data_bundle.get("balanced_reference_classes"),
            "class_balance_adjustments": data_bundle.get("class_balance_adjustments"),
            "train_split": data_bundle.get("train_split"),
            "reference_split": data_bundle.get("reference_split")
            or getattr(config.data, "reference_split", None),
            "train_role": data_bundle.get("train_role"),
            "reference_role": data_bundle.get("reference_role"),
            "train_source_id": data_bundle.get("train_source_id"),
            "reference_source_id": data_bundle.get("reference_source_id"),
            "data_validation": data_bundle.get("data_validation"),
            "batching": batching,
            "runtime": runtime,
            "seed_policy": seed_policy,
            "resume": resume_metadata,
        }
        save_json(run_dir / "artifacts" / "data_meta.json", data_meta)
        data_summary = _data_summary(
            config,
            data_bundle,
            train_tensor,
            reference_tensor,
            batching,
        )
        tracker.stage(
            "data_prepared",
            message=(
                "data prepared "
                f"train_shape={list(train_tensor.shape)} "
                f"reference_shape={list(reference_tensor.shape)} "
                f"batch_cap={batching['batch_cap']} "
                f"effective_batch_size={batching['effective_batch_size']} "
                f"drop_last={batching['drop_last']}"
            ),
            data_meta_path=str((run_dir / "artifacts" / "data_meta.json").resolve()),
            batching=batching,
            runtime=runtime,
            seed_policy=seed_policy,
            data_summary=data_summary,
        )
        _log_data_summary(tracker, data_summary)

        generation_monitor = {
            "z_init_path": str((run_dir / "artifacts" / "z_init.pt").resolve()),
            "monitor_batch_size": int(_configured_monitor_batch_size(config)),
        }
        train_grid_path = _save_generation_monitor_train_grid(
            config,
            train_tensor,
            run_dir,
            tracker,
        )
        if train_grid_path is not None:
            generation_monitor["train_grid_path"] = str(Path(train_grid_path).resolve())
        checkpoint_trace_samples = generation_monitor.setdefault("checkpoint_samples", [])
        random_projected_f_mem = _prepare_random_projected_f_mem(
            config,
            train_tensor,
            run_dir,
            tracker,
            device,
        )

        # -----------------------------------------------------------------
        # Phase 5: build the model and optimizer.
        #
        # GDFM reads the config to choose the prediction/loss spaces and to
        # build the configured backbone: MLP, UNet, or JiT.
        # -----------------------------------------------------------------
        model = GDFM(config).to(device)
        model_summary = _model_summary(config, model)
        tracker.stage(
            "model_created",
            message=(
                "model created "
                f"backbone={model_summary['backbone']} "
                f"arch={model_summary['network_arch']} "
                f"prediction_space={model_summary['prediction_space']} "
                f"loss_space={model_summary['loss_space']} "
                f"trainable_parameters={model_summary['trainable_parameters']:,}"
            ),
            model_summary=model_summary,
        )
        _log_model_summary(tracker, model, model_summary)
        optimizer = _build_optimizer(config, model)
        start_step = 1
        resume_checkpoint_path = None
        if resume_source is not None:
            resume_checkpoint_path = _link_resume_checkpoint(run_dir, resume_source)
            checkpoint_state = torch.load(
                resume_source["source_checkpoint"],
                map_location=device,
            )
            payload_step = int(checkpoint_state.get("step", resume_source["source_step"]))
            if payload_step != int(resume_source["source_step"]):
                raise ValueError(
                    "Resume checkpoint payload step does not match filename: "
                    f"payload={payload_step} filename={resume_source['source_step']}"
                )
            model.load_state_dict(checkpoint_state["model"])
            optimizer.load_state_dict(checkpoint_state["optimizer"])
            start_step = payload_step + 1
            tracker.checkpoint_saved(
                step=payload_step,
                checkpoint_path=resume_checkpoint_path,
            )
            tracker.note(
                (
                    "loaded resume checkpoint "
                    f"step={payload_step} -> {Path(resume_checkpoint_path).resolve()}"
                ),
                event_type="resume_checkpoint_loaded",
                resume=resume_metadata,
            )

        # This generator controls the DataLoader shuffle order only. The
        # model's internal loss still samples its own times/noise unless
        # callers pass explicit values into `compute_loss`.
        generator = torch.Generator().manual_seed(int(config.training.seed))
        train_loader = _build_train_loader(
            train_tensor,
            batching,
            generator,
            pin_memory=bool(runtime["pin_memory"]),
        )
        train_iterator = iter(train_loader)

        # -----------------------------------------------------------------
        # Phase 6: train and checkpoint.
        # -----------------------------------------------------------------
        history = _load_resume_history(resume_source)

        # Save step 0 before any optimizer updates so evaluation can measure
        # the untrained model under the same path as later checkpoints.
        if resume_source is None:
            checkpoint_path = save_checkpoint(run_dir, 0, model, optimizer)
            tracker.checkpoint_saved(step=0, checkpoint_path=checkpoint_path)
            try:
                trace_record = _save_generation_monitor_checkpoint_samples(
                    config,
                    model,
                    train_tensor,
                    z_init,
                    run_dir,
                    device,
                    tracker,
                    step=0,
                )
                if trace_record is not None:
                    checkpoint_trace_samples.append(trace_record)
            except Exception as exc:
                tracker.warn(
                    "generation monitor sample grid skipped "
                    f"step=0: {type(exc).__name__}: {exc}"
                )
        train_start = time.perf_counter()
        tracker.stage(
            "training",
            message=(
                "training started "
                f"steps={int(start_step)}..{int(target_step)} "
                f"logical_steps={int(config.training.n_training_steps)} "
                f"checkpoints={len(checkpoint_steps)} "
                f"batch_cap={batching['batch_cap']} "
                f"effective_batch_size={batching['effective_batch_size']} "
                f"drop_last={batching['drop_last']} "
                f"device={device}"
            ),
            batching=batching,
            runtime=runtime,
            checkpoint_schedule=checkpoint_schedule,
        )
        progress = tqdm(
            range(int(start_step), int(target_step) + 1),
            desc=f"train:{config.outputs.run_name}",
            unit="step",
            disable=not tracker.interactive_progress,
            dynamic_ncols=True,
        )

        try:
            for step in progress:
                # Draw from the DataLoader over the fixed training tensor, then
                # move only the batch to the target device. Keeping the stored
                # tensor on CPU avoids pinning the whole dataset in GPU memory.
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                batch = batch.to(
                    device,
                    non_blocking=bool(runtime["non_blocking_h2d"]),
                )

                # The whole learning rule is these four lines: clear old
                # gradients, compute the GDFM loss, backpropagate, and update
                # parameters.
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(config, device):
                    loss = model.compute_loss(batch)
                loss.backward()
                optimizer.step()

                # Keep history sparse. For long runs we do not need one JSON
                # record per step; we need enough points for progress, plots,
                # and final summaries.
                if (
                    step == 1
                    or step % int(config.training.log_every) == 0
                    or step == int(target_step)
                ):
                    loss_value = float(loss.item())
                    history.append({"step": int(step), "loss": loss_value})
                    save_json(run_dir / "metrics" / "train_history.json", history)
                    if not progress.disable:
                        progress.set_postfix(loss=f"{loss_value:.4f}", refresh=False)
                    tracker.training_progress(
                        step=int(step),
                        total_steps=int(config.training.n_training_steps),
                        loss=loss_value,
                        elapsed_sec=time.perf_counter() - train_start,
                        force=step == 1 or step == int(target_step),
                    )

                # Checkpoints include both model and optimizer state. The final
                # checkpoint is always saved even if `n_steps` is not divisible
                # by `ckpt_every`.
                if int(step) in checkpoint_step_set:
                    checkpoint_path = save_checkpoint(run_dir, step, model, optimizer)
                    tracker.checkpoint_saved(step=step, checkpoint_path=checkpoint_path)
                    try:
                        trace_record = _save_generation_monitor_checkpoint_samples(
                            config,
                            model,
                            train_tensor,
                            z_init,
                            run_dir,
                            device,
                            tracker,
                            step=step,
                        )
                        if trace_record is not None:
                            checkpoint_trace_samples.append(trace_record)
                    except Exception as exc:
                        tracker.warn(
                            "generation monitor sample grid skipped "
                            f"step={int(step)}: {type(exc).__name__}: {exc}"
                        )
                    try:
                        _run_random_projected_f_mem(
                            config,
                            model,
                            train_tensor,
                            z_init,
                            random_projected_f_mem,
                            device,
                            tracker,
                            step=step,
                        )
                    except Exception as exc:
                        tracker.warn(
                            "projected f_mem monitor skipped "
                            f"step={int(step)}: {type(exc).__name__}: {exc}"
                        )
        finally:
            # Close tqdm cleanly even when training is interrupted or errors.
            progress.close()

        training_wall_time_sec = time.perf_counter() - train_start
        total_wall_time_sec = time.perf_counter() - run_start
        logical_n_steps = int(config.training.n_training_steps)
        completed_step = int(target_step)
        executed_steps = max(1, int(target_step) - int(start_step) + 1)
        stop_after_step = (
            None
            if getattr(config.training, "stop_after_step", None) in (None, "")
            else int(config.training.stop_after_step)
        )

        # -----------------------------------------------------------------
        # Phase 7: write final summaries.
        #
        # `train_history.json` is the machine-readable loss curve.
        # `run_summary.json` is the quick operator-facing overview.
        # -----------------------------------------------------------------
        save_json(run_dir / "metrics" / "train_history.json", history)
        training_figure_path = _save_training_loss_figure(history, run_dir / "figures")
        generated_figures = dict(tracker.status.get("generated_figures") or {})
        if training_figure_path is not None:
            tracker.register_figure("training_loss_vs_step", training_figure_path)
            generated_figures = dict(tracker.status.get("generated_figures") or {})
        final_trace_path = _save_final_generation_monitor_samples(
            config,
            model,
            train_tensor,
            z_init,
            run_dir,
            device,
            tracker,
        )
        if final_trace_path is not None:
            generation_monitor["final_samples_path"] = str(
                Path(final_trace_path).resolve()
            )
            generated_figures = dict(tracker.status.get("generated_figures") or {})
        random_projected_f_mem_summary = {
            "enabled": bool(random_projected_f_mem.get("enabled", False)),
            "metadata": random_projected_f_mem.get("metadata"),
            "records": list(random_projected_f_mem.get("records", [])),
        }
        if bool(random_projected_f_mem.get("enabled", False)):
            save_json(
                run_dir / "metrics" / "random_projected_f_mem.json",
                random_projected_f_mem.get("records", []),
            )

        run_summary = {
            "study": config.study.name,
            "run_dir": str(run_dir),
            "experiment_dir": str(study_dir.resolve()),
            "run_id": run_dir.parents[3].name,
            "seed_id": run_dir.parents[1].name,
            "attempt_id": run_dir.name,
            "seed": int(config.training.seed),
            "n_training_steps": logical_n_steps,
            "start_step": int(start_step),
            "completed_step": completed_step,
            "executed_steps": executed_steps,
            "logical_target_step": logical_n_steps,
            "stop_after_step": stop_after_step,
            "resume": resume_metadata,
            "checkpoint_schedule": checkpoint_schedule,
            "n_train": int(config.data.n_train),
            "n_test": int(config.data.n_test),
            "batching": batching,
            "data_summary": data_summary,
            "batch_cap": int(batching["batch_cap"]),
            "effective_batch_size": int(batching["effective_batch_size"]),
            "drop_last": bool(batching["drop_last"]),
            "optimizer": str(config.training.optimizer),
            "lr": float(config.training.lr),
            "weight_decay": float(config.training.weight_decay),
            "momentum": float(config.training.momentum),
            "model_summary": model_summary,
            "runtime": runtime,
            "seed_policy": seed_policy,
            "training_wall_time_sec": training_wall_time_sec,
            "training_seconds_per_step": (
                training_wall_time_sec / executed_steps
            ),
            "total_wall_time_sec": total_wall_time_sec,
            "figures_dir": str((run_dir / "figures").resolve()),
            "logs_dir": str((run_dir / "logs").resolve()),
            "status_path": str((run_dir / "status.json").resolve()),
            "attempt_log_path": str((run_dir / "logs" / "attempt.log").resolve()),
            "generated_figures": generated_figures,
            "generation_monitor": generation_monitor,
            "random_projected_f_mem": random_projected_f_mem_summary,
            "wandb": tracker.status.get("wandb"),
        }
        save_json(run_dir / "artifacts" / "run_summary.json", run_summary)
        tracker.log_metrics(
            {
                "train/final_loss": history[-1]["loss"] if history else None,
                "train/training_wall_time_sec": training_wall_time_sec,
                "train/training_seconds_per_step": (
                    training_wall_time_sec / executed_steps
                ),
            },
            step=completed_step,
        )

        # The tracker stage marks training complete, but not full evaluation
        # complete. `evaluate_run(...)` later transitions the same attempt to a
        # completed state after checkpoint metrics are written.
        tracker.stage(
            "trained",
            message=(
                "training complete "
                f"wall_time={training_wall_time_sec:.2f}s "
                f"seconds_per_step={training_wall_time_sec / executed_steps:.4f} "
                f"completed_step={completed_step}/{logical_n_steps}"
            ),
            run_summary_path=str((run_dir / "artifacts" / "run_summary.json").resolve()),
            train_history_path=str((run_dir / "metrics" / "train_history.json").resolve()),
            generated_figures=generated_figures,
            generation_monitor=generation_monitor,
            random_projected_f_mem=random_projected_f_mem_summary,
            runtime=runtime,
            seed_policy=seed_policy,
            checkpoint_schedule=checkpoint_schedule,
            resume=resume_metadata,
        )
        run_dir = mark_attempt_done(run_dir)
        tracker.attempt_dir = run_dir
        tracker.status = load_attempt_status(run_dir)
        tracker.note(
            f"attempt folder marked done path={run_dir.resolve()}",
            event_type="attempt_folder_marked_done",
            original_attempt_id=tracker.status.get("original_attempt_id"),
        )
        return run_dir
    except KeyboardInterrupt as exc:
        # Preserve an intentional interruption as an explicit attempt state so
        # dashboards and manifests do not confuse it with a successful run.
        if tracker is not None:
            tracker.stage("aborted", last_error=f"{type(exc).__name__}: {exc}")
        else:
            update_attempt_status(run_dir, "aborted", last_error=f"{type(exc).__name__}: {exc}")
        raise
    except Exception as exc:
        # Any unexpected error should still leave the attempt directory
        # inspectable, with status.json pointing at the failure.
        if tracker is not None:
            tracker.stage("failed", last_error=f"{type(exc).__name__}: {exc}")
        else:
            update_attempt_status(run_dir, "failed", last_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # AttemptTracker may own file handles, console tee state, or external
        # logging handles. Close it regardless of success or failure.
        if tracker is not None:
            tracker.close()
        _release_resume_lock(resume_lock_handle)


if __name__ == "__main__":
    import tempfile

    from mp_config import get_base_config

    print("mp_training.trainer example")

    # Build the smallest end-to-end training config that still writes a real run.
    config = get_base_config()

    # Study/output names determine the human-facing run labels and become part
    # of the artifact folder names.
    config.study.name = "trainer_demo"
    config.outputs.run_name = "trainer_demo"

    # Use the synthetic GMM path so the example does not need image files on
    # disk. These fields define the size of the ambient space and planted
    # latent mixture.
    config.data.dataset = "gmm_subspace"
    config.data.n_train = 32
    config.data.n_test = 16
    config.data.data_dim = 8
    config.data.intrinsic_dim = 2
    config.data.n_clusters = 3

    # Pick the tiny MLP backbone. The same `GDFM(config)` call would build a
    # UNet or JiT backbone if `config.network.arch` requested one.
    config.network.arch = "mlp"
    config.network.hidden_dim = 32
    config.network.n_layers = 2
    config.network.time_emb_dim = 16

    # Keep the training schedule short enough for the example while still
    # producing multiple checkpoints and multiple loss-history entries.
    config.training.n_training_steps = 4
    config.training.batch_cap = 8
    config.training.batch_size = 8
    config.training.ckpt_every = 2
    config.training.log_every = 1

    # These evaluation fields are saved during training because checkpoint
    # evaluation later needs a fixed sample count, panel count, and distance chunk.
    config.evaluation.n_samples = 8
    config.evaluation.sample_panel_count = 4
    config.evaluation.chunk_size = 32
    config.sampling.n_steps = 4

    # Show the one-line training entrypoint before running the tiny example.
    print("Usage:")
    print("  config = get_base_config()")
    print("  run_dir = train_run(config, study_dir, device='cpu')")

    # Run into a temporary directory so the example is safe to execute anywhere.
    with tempfile.TemporaryDirectory() as tmpdir:
        # This is the same call used by experiments and canonical runners:
        # pass a resolved config, choose an experiment root, and pick a device.
        run_dir = train_run(config, Path(tmpdir) / "trainer_demo", device="cpu")

        # These paths are the key outputs to inspect after a training run.
        print(f"run_dir              : {run_dir}")
        print(f"checkpoints dir      : {run_dir / 'checkpoints'}")
        print(f"metrics file         : {run_dir / 'metrics' / 'train_history.json'}")
