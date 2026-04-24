"""
Image checkpoint evaluation for saved image-runner attempts.

This module loads saved run artifacts, samples every checkpoint from the saved
`z_init`, computes image metrics, saves preview artifacts, and updates tracking.
The metric definitions stay in `mp_metrics.metrics_images`.
"""

from pathlib import Path

import torch

from gdfm import GDFM
from mp_artifacts import (
    list_checkpoint_paths,
    load_config_snapshot,
    load_json,
    save_json,
    save_tensor,
)
from mp_metrics.metrics_images import (
    RHO_SWEEP,
    collect_image32_metric_record,
    compute_memorization_metrics,
    compute_nn_stats,
)
from mp_metrics.loss_eval import collect_loss_eval_record
from mp_runtime import autocast_context, configure_torch_runtime
from mp_tracking import AttemptTracker, loss_eval_metric_payload


DEFAULT_CHECKPOINT_PREVIEW_COUNT = 8


def checkpoint_preview_figure_name(step):
    """
    Return the stable figure key for all checkpoint preview panels.
    """
    return "checkpoint_samples_over_train"


def checkpoint_preview_figure_path(step, run_dir):
    """
    Return the on-disk path for the current checkpoint preview panel.

    We keep a single rotating panel file while logging progression in W&B
    via `eval/checkpoint_samples_over_train` with explicit step metadata.
    """
    return Path(run_dir) / "figures" / "checkpoint_samples_over_train_latest.png"


def _generation_monitor_config(config):
    """Return enabled generation monitor config, if present."""
    trace = getattr(config, "generation_monitor", None) if config is not None else None
    if trace is None or not bool(getattr(trace, "enabled", False)):
        return None
    return trace


def _preview_count(config, samples, train_tensor=None, count=None):
    """Resolve how many checkpoint samples to save."""
    if count is None:
        trace = _generation_monitor_config(config)
        if trace is not None:
            count = int(getattr(trace, "n_samples", DEFAULT_CHECKPOINT_PREVIEW_COUNT))
        else:
            evaluation = getattr(config, "evaluation", None) if config is not None else None
            count = int(
                getattr(
                    evaluation,
                    "sample_panel_count",
                    DEFAULT_CHECKPOINT_PREVIEW_COUNT,
                )
            )

    limit = int(samples.shape[0])
    if train_tensor is not None:
        limit = min(limit, int(train_tensor.shape[0]))
    return max(0, min(int(count), limit))


def collect_checkpoint_sample_preview_record(
    *,
    step,
    samples,
    train_tensor,
    run_dir,
    config=None,
    count=None,
    figure_enabled=None,
):
    """
    Save checkpoint samples and, when requested, a samples-over-train panel.

    The tensor payload is always saved. The image figure is rendered only when
    tensors are image-shaped and generation monitoring is enabled.
    """
    from mp_visualization import (
        is_image_batch,
        model_context_title,
        save_samples_over_train_panel,
    )

    step = int(step)
    run_dir = Path(run_dir)
    count = _preview_count(config, samples, train_tensor=train_tensor, count=count)
    samples_dir = run_dir / "metrics" / "checkpoint_sample_previews"
    samples_path = samples_dir / f"step_{step:06d}.pt"
    samples_dir.mkdir(parents=True, exist_ok=True)

    preview_samples = samples[:count].detach().cpu()
    torch.save(preview_samples, samples_path)

    figure_path = None
    should_render = (
        bool(figure_enabled)
        if figure_enabled is not None
        else _generation_monitor_config(config) is not None
    )
    if (
        should_render
        and count > 0
        and train_tensor is not None
        and is_image_batch(preview_samples)
        and is_image_batch(train_tensor)
    ):
        figure_name = checkpoint_preview_figure_name(step)
        saved_path = save_samples_over_train_panel(
            preview_samples,
            train_tensor,
            checkpoint_preview_figure_path(step, run_dir),
            count=count,
            title=(
                model_context_title(
                    config,
                    "Checkpoint samples over train images",
                    checkpoint_step=step,
                )
                if config is not None
                else f"Checkpoint samples over train images\ncheckpoint_step={step}"
            ),
        )
        if saved_path is not None:
            figure_path = str(Path(saved_path).resolve())

    return {
        "step": step,
        "count": int(count),
        "samples_path": str(samples_path.resolve()),
        "figure_path": figure_path,
    }


def collect_reference_memory(reference_tensor, train_tensor, *, chunk_size, config):
    """
    Measure held-out reference samples against the training set.

    This is a pre-training baseline for the image runner and a leakage check for
    the selected train/reference split.
    """
    count = min(int(config.evaluation.n_samples), int(reference_tensor.shape[0]))
    return compute_memorization_metrics(
        reference_tensor[:count],
        train_tensor,
        chunk_size=int(chunk_size),
        config=config,
    )


def image_metric_payload(record):
    """Return tracker metric keys for one image checkpoint record."""
    payload = {
        "eval/f_mem": record["f_mem"],
        "eval/baseline_f_mem": record["baseline_f_mem"],
        "eval/delta_f_mem": record["delta_f_mem"],
        "eval/mean_r": record["mean_r"],
        "eval/median_r": record["median_r"],
        "eval/baseline_mean_r": record["baseline_mean_r"],
        "eval/baseline_median_r": record["baseline_median_r"],
    }
    for rho_val in RHO_SWEEP:
        suffix = f"rho{rho_val}"
        payload[f"eval/f_mem_{suffix}"] = record[f"f_mem_{suffix}"]
        payload[f"eval/baseline_f_mem_{suffix}"] = record[
            f"baseline_f_mem_{suffix}"
        ]
        payload[f"eval/delta_f_mem_{suffix}"] = record[f"delta_f_mem_{suffix}"]
    return payload


def evaluate_image32_run(run_dir, device="cpu", fixed_times=None):
    """
    Replay image checkpoint evaluation for one saved attempt directory.

    The evaluator loads saved train/reference tensors and saved `z_init`, then
    samples every checkpoint from the same initial noise.
    """
    run_dir = Path(run_dir)
    config = load_config_snapshot(run_dir)
    if fixed_times is None:
        fixed_times = tuple(config.evaluation.loss_eval_times)
    configure_torch_runtime(config, device)
    tracker = None
    try:
        tracker = AttemptTracker(config, run_dir)
        tracker.stage("image32_evaluating", message=f"Image32 evaluation started device={device}")

        train_tensor = torch.load(run_dir / "artifacts" / "train_tensor.pt", map_location="cpu")
        reference_tensor = torch.load(
            run_dir / "artifacts" / "reference_tensor.pt",
            map_location="cpu",
        )
        z_init = torch.load(run_dir / "artifacts" / "z_init.pt", map_location="cpu")

        model = GDFM(config).to(device)
        records = []
        loss_records = []
        checkpoint_sample_previews = []
        checkpoint_paths = list_checkpoint_paths(run_dir)
        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

        for checkpoint_path in checkpoint_paths:
            # The saved z_init is reused at every checkpoint for fair replay.
            state = torch.load(checkpoint_path, map_location=device)
            step = int(state["step"])
            model.load_state_dict(state["model"])
            with autocast_context(config, device):
                samples = model.sample(
                    z_init.clone().to(device),
                    n_steps=int(config.sampling.n_steps),
                    method=str(config.sampling.method),
                    device=device,
                ).cpu()

            record = collect_image32_metric_record(
                step,
                samples,
                reference_tensor,
                train_tensor,
                chunk_size=int(config.evaluation.chunk_size),
                config=config,
            )
            loss_record = collect_loss_eval_record(
                step,
                model,
                reference_tensor,
                config,
                device,
                fixed_times=fixed_times,
            )
            records.append(record)
            loss_records.append(loss_record)
            metric_payload = image_metric_payload(record)
            metric_payload.update(loss_eval_metric_payload(loss_record))
            tracker.log_metrics(metric_payload, step=step)

            try:
                preview_record = collect_checkpoint_sample_preview_record(
                    step=step,
                    samples=samples,
                    train_tensor=train_tensor,
                    run_dir=run_dir,
                    config=config,
                )
                checkpoint_sample_previews.append(preview_record)
                if preview_record.get("figure_path"):
                    figure_name = checkpoint_preview_figure_name(step)
                    tracker.register_figure(
                        figure_name,
                        preview_record["figure_path"],
                        log_to_wandb=False,
                    )
                    tracker.log_figure_metric(
                        "eval/checkpoint_samples_over_train",
                        preview_record["figure_path"],
                        step=step,
                    )
            except Exception as exc:
                tracker.warn(
                    "checkpoint sample preview skipped "
                    f"step={step}: {type(exc).__name__}: {exc}"
                )

            if checkpoint_path == checkpoint_paths[-1]:
                neighbor_stats = compute_nn_stats(
                    samples,
                    train_tensor,
                    chunk_size=int(config.evaluation.chunk_size),
                )
                panel_count = min(int(config.evaluation.sample_panel_count), samples.shape[0])
                save_tensor(run_dir / "metrics" / "final_samples.pt", samples[:panel_count])
                save_tensor(
                    run_dir / "metrics" / "final_neighbors.pt",
                    train_tensor[neighbor_stats["nn_idx"][:panel_count]],
                )

        save_json(run_dir / "metrics" / "checkpoint_metrics.json", records)
        save_json(run_dir / "metrics" / "loss_eval.json", loss_records)
        save_json(
            run_dir / "metrics" / "checkpoint_sample_previews.json",
            checkpoint_sample_previews,
        )
        _update_run_summary(run_dir, records, loss_records, checkpoint_sample_previews)
        tracker.stage(
            "completed",
            message=f"Image32 evaluation complete checkpoints={len(records)}",
            checkpoint_sample_previews_path=str(
                (run_dir / "metrics" / "checkpoint_sample_previews.json").resolve()
            ),
            checkpoint_sample_previews=checkpoint_sample_previews,
        )
        return {
            "checkpoint_metrics": records,
            "loss_eval": loss_records,
            "checkpoint_sample_previews": checkpoint_sample_previews,
        }
    except Exception as exc:
        if tracker is not None:
            tracker.stage("failed", last_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if tracker is not None:
            tracker.close()


def _update_run_summary(run_dir, records, loss_records, checkpoint_sample_previews=None):
    """Write image evaluation paths and final metrics into run summary."""
    run_dir = Path(run_dir)
    run_summary_path = run_dir / "artifacts" / "run_summary.json"
    run_summary = load_json(run_summary_path) if run_summary_path.exists() else {}
    run_summary.update(
        {
            "checkpoint_metrics_path": str(
                (run_dir / "metrics" / "checkpoint_metrics.json").resolve()
            ),
            "loss_eval_path": str((run_dir / "metrics" / "loss_eval.json").resolve()),
            "checkpoint_sample_previews_path": str(
                (run_dir / "metrics" / "checkpoint_sample_previews.json").resolve()
            ),
            "checkpoint_sample_previews": list(checkpoint_sample_previews or []),
            "final_samples_path": str((run_dir / "metrics" / "final_samples.pt").resolve()),
            "final_neighbors_path": str((run_dir / "metrics" / "final_neighbors.pt").resolve()),
            "final_metrics": records[-1] if records else {},
            "final_loss_eval": loss_records[-1] if loss_records else {},
        }
    )
    save_json(run_summary_path, run_summary)


__all__ = [
    "DEFAULT_CHECKPOINT_PREVIEW_COUNT",
    "checkpoint_preview_figure_name",
    "collect_checkpoint_sample_preview_record",
    "collect_reference_memory",
    "evaluate_image32_run",
    "image_metric_payload",
]


if __name__ == "__main__":
    print("mp_runners.image_eval example")
    print("preview key:", checkpoint_preview_figure_name(100))
