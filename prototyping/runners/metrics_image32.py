"""
Empirical Image32 memorization and loss-evaluation helpers.

These helpers keep image-runner metrics behind intent-named functions. They do
not build configs, choose datasets, or train models.
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
from mp_metrics import compute_memorization_metrics
from mp_metrics.checkpoint_previews import (
    checkpoint_preview_figure_name,
    collect_checkpoint_sample_preview_record,
)
from mp_metrics.loss_eval import (
    DEFAULT_LOSS_EVAL_TIMES,
    collect_loss_eval_record,
    loss_eval_metric_payload,
)
from mp_metrics.memorization import compute_nn_stats
from mp_runtime import autocast_context, configure_torch_runtime
from mp_tracking import AttemptTracker


DEFAULT_FIXED_TIMES = DEFAULT_LOSS_EVAL_TIMES


def collect_image32_metric_record(
    step,
    samples,
    baseline_samples,
    train_tensor,
    *,
    chunk_size,
    config,
):
    """Collect generated-sample and held-out-reference memorization metrics."""
    generated = compute_memorization_metrics(
        samples,
        train_tensor,
        chunk_size=int(chunk_size),
        config=config,
    )
    baseline = compute_memorization_metrics(
        baseline_samples[: samples.shape[0]],
        train_tensor,
        chunk_size=int(chunk_size),
        config=config,
    )

    record = {
        "step": int(step),
        "ratio_threshold": generated["ratio_threshold"],
        "f_mem": generated["f_mem"],
        "mean_nn_ratio": generated["mean_nn_ratio"],
        "std_nn_ratio": generated["std_nn_ratio"],
        "mean_d1": generated["mean_d1"],
        "mean_d2": generated["mean_d2"],
        "baseline_f_mem": baseline["f_mem"],
        "baseline_mean_nn_ratio": baseline["mean_nn_ratio"],
        "baseline_std_nn_ratio": baseline["std_nn_ratio"],
        "baseline_mean_d1": baseline["mean_d1"],
        "baseline_mean_d2": baseline["mean_d2"],
    }
    record["nn_ratio"] = record["mean_nn_ratio"]
    record["baseline_nn_ratio"] = record["baseline_mean_nn_ratio"]
    return record


def collect_reference_memory(reference_tensor, train_tensor, *, chunk_size, config):
    """Measure whether a held-out reference subset already looks train-like."""
    count = min(int(config.evaluation.n_samples), int(reference_tensor.shape[0]))
    return compute_memorization_metrics(
        reference_tensor[:count],
        train_tensor,
        chunk_size=int(chunk_size),
        config=config,
    )


def evaluate_image32_run(run_dir, device="cpu", fixed_times=None):
    """Evaluate checkpoint memorization and fixed/random-t losses."""
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
            tracker.log_metrics(
                {
                    "eval/f_mem": record["f_mem"],
                    "eval/mean_nn_ratio": record["mean_nn_ratio"],
                    "eval/std_nn_ratio": record["std_nn_ratio"],
                    "eval/baseline_f_mem": record["baseline_f_mem"],
                    "eval/baseline_mean_nn_ratio": record["baseline_mean_nn_ratio"],
                    "eval/baseline_std_nn_ratio": record["baseline_std_nn_ratio"],
                    **loss_eval_metric_payload(loss_record),
                },
                step=step,
            )

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
                    train_tensor[neighbor_stats["indices"][:panel_count]],
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
    "DEFAULT_FIXED_TIMES",
    "collect_image32_metric_record",
    "collect_loss_eval_record",
    "collect_reference_memory",
    "evaluate_image32_run",
]
