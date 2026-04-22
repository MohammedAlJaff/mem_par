"""
Generic checkpoint replay workflow for saved runs.

This evaluator is the simple tensor baseline for saved attempts. Specialized
runners can use their own replay functions when they need dataset-specific
geometry or image figures.
"""

import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from gdfm import GDFM
from mp_artifacts import (
    list_checkpoint_paths,
    load_config_snapshot,
    load_json,
    save_json,
    save_tensor,
    update_attempt_status,
)
from mp_runners.image_eval import (
    checkpoint_preview_figure_name,
    collect_checkpoint_sample_preview_record,
)
from mp_metrics.loss_eval import collect_loss_eval_record
from mp_metrics.synthetic import compute_nn_stats
from mp_runtime import autocast_context, configure_torch_runtime
from mp_tracking import AttemptTracker, loss_eval_metric_payload



def _validate_saved_holdout_metadata(run_dir):
    """
    Re-check saved train/reference source metadata before checkpoint evaluation.

    Evaluation should never silently compare generated samples against a
    reference tensor that overlaps the training tensor.
    """
    run_dir = Path(run_dir)
    data_meta_path = run_dir / "artifacts" / "data_meta.json"
    if not data_meta_path.exists():
        return

    data_meta = load_json(data_meta_path)
    train_source_id = data_meta.get("train_source_id")
    reference_source_id = data_meta.get("reference_source_id")
    if train_source_id is None or reference_source_id is None:
        return

    if str(train_source_id) != str(reference_source_id):
        return

    train_indices_path = run_dir / "artifacts" / "train_indices.pt"
    reference_indices_path = run_dir / "artifacts" / "reference_indices.pt"
    if not train_indices_path.exists() or not reference_indices_path.exists():
        raise ValueError(
            "Data leakage guard failed during evaluation: train/reference share "
            f"source {train_source_id!r}, but saved source indices are missing."
        )

    train_indices = set(
        int(item)
        for item in torch.load(train_indices_path, map_location="cpu").flatten().tolist()
    )
    reference_indices = set(
        int(item)
        for item in torch.load(reference_indices_path, map_location="cpu").flatten().tolist()
    )
    overlap = train_indices.intersection(reference_indices)
    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(
            "Data leakage guard failed during evaluation: saved train/reference "
            f"indices overlap for source {train_source_id!r}. "
            f"overlap_count={len(overlap)}, examples={examples}"
        )

    validation = data_meta.get("data_validation") or {}
    if not bool(validation.get("same_source_allowed", False)):
        raise ValueError(
            "Data leakage guard failed during evaluation: train/reference share "
            f"source {train_source_id!r}. This run was not marked as an allowed "
            "same-source disjoint holdout."
        )


def _validate_saved_tensor_identity(train_tensor, reference_tensor):
    """
    Fail if saved train/reference tensors are exactly identical full batches.
    """
    if train_tensor.shape == reference_tensor.shape and torch.equal(
        train_tensor,
        reference_tensor,
    ):
        raise ValueError(
            "Data leakage guard failed during evaluation: train and reference "
            "tensors are exactly equal."
        )


def _save_checkpoint_metric_figures(records, figures_dir):
    """
    Save the standard checkpoint-evaluation scalar figures for one attempt.
    """
    if not records:
        return {}

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(record["step"]) for record in records]

    figure_paths = {}

    f_mem_path = figures_dir / "f_mem_vs_tau.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [record["f_mem"] for record in records], marker="o", label="generated")
    ax.plot(
        steps,
        [record["baseline_f_mem"] for record in records],
        marker="o",
        linestyle="--",
        label="held-out reference",
    )
    ax.set_title("Memorization fraction over training")
    ax.set_xlabel("checkpoint step tau")
    ax.set_ylabel("f_mem")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f_mem_path, dpi=160)
    plt.close(fig)
    figure_paths["f_mem_vs_tau"] = f_mem_path

    mean_r_path = figures_dir / "mean_r_vs_tau.png"
    means = [record["mean_r"] for record in records]
    medians = [record["median_r"] for record in records]
    baseline_means = [record["baseline_mean_r"] for record in records]
    baseline_medians = [record["baseline_median_r"] for record in records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, means, marker="o", label="generated mean")
    ax.plot(steps, medians, marker="o", linestyle=":", label="generated median")
    ax.plot(steps, baseline_means, marker="o", linestyle="--", label="held-out mean")
    ax.plot(steps, baseline_medians, marker="o", linestyle="-.", label="held-out median")
    ax.set_title("Nearest-neighbor ratio over training")
    ax.set_xlabel("checkpoint step tau")
    ax.set_ylabel("r")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(mean_r_path, dpi=160)
    plt.close(fig)
    figure_paths["mean_r_vs_tau"] = mean_r_path

    return figure_paths


def evaluate_run(run_dir, device="cpu"):
    """
    Evaluate every checkpoint in a saved run directory and collect metrics.

    For each checkpoint we:
        1. reload model weights
        2. regenerate samples from the saved shared `z_init`
        3. compare those samples against the saved training tensor
        4. compare held-out reference samples against the same training tensor
           as a baseline

    On the final checkpoint we also save a small sample/nearest-neighbor panel
    payload so later plotting code can show concrete examples in addition to
    scalar curves.
    """
    run_dir = Path(run_dir)
    tracker = None
    try:
        config = load_config_snapshot(run_dir)
        configure_torch_runtime(config, device)
        tracker = AttemptTracker(config, run_dir)
        tracker.stage("evaluating", message=f"checkpoint evaluation started device={device}")
        _validate_saved_holdout_metadata(run_dir)

        train_tensor = torch.load(
            run_dir / "artifacts" / "train_tensor.pt",
            map_location="cpu",
        )
        reference_tensor = torch.load(
            run_dir / "artifacts" / "reference_tensor.pt",
            map_location="cpu",
        )
        _validate_saved_tensor_identity(train_tensor, reference_tensor)
        z_init = torch.load(run_dir / "artifacts" / "z_init.pt", map_location="cpu")

        model = GDFM(config).to(device)
        records = []
        loss_records = []
        checkpoint_sample_previews = []
        checkpoint_paths = list_checkpoint_paths(run_dir)

        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

        eval_start = time.perf_counter()
        progress = tqdm(
            checkpoint_paths,
            desc=f"eval:{config.outputs.run_name}",
            unit="ckpt",
            disable=not tracker.interactive_progress,
            dynamic_ncols=True,
        )

        try:
            for checkpoint_index, checkpoint_path in enumerate(progress, start=1):
                # Use the same saved `z_init` at every checkpoint so changes in the
                # curve reflect the model, not evaluation randomness.
                state = torch.load(checkpoint_path, map_location=device)
                step = int(state["step"])
                model.load_state_dict(state["model"])
                loss_record = collect_loss_eval_record(
                    step,
                    model,
                    reference_tensor,
                    config,
                    device,
                )
                loss_records.append(loss_record)
                tracker.log_metrics(loss_eval_metric_payload(loss_record), step=step)
                with autocast_context(config, device):
                    samples = model.sample(
                        z_init.clone().to(device),
                        n_steps=int(config.sampling.n_steps),
                        method=str(config.sampling.method),
                        device=device,
                    ).cpu()
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

                rho = float(config.evaluation.ratio_threshold)
                generated_stats = compute_nn_stats(
                    samples,
                    train_tensor,
                    chunk_size=int(config.evaluation.chunk_size),
                )
                baseline_tensor = reference_tensor[: samples.shape[0]]
                baseline_stats = compute_nn_stats(
                    baseline_tensor,
                    train_tensor,
                    chunk_size=int(config.evaluation.chunk_size),
                )
                r_gen = generated_stats["r"]  # r(x) = d1(x) / d2(x), generated samples
                r_base = baseline_stats["r"]  # r(x) = d1(x) / d2(x), held-out baseline
                f_mem = float((r_gen < rho).float().mean().item())
                baseline_f_mem = float((r_base < rho).float().mean().item())

                # Keep both generated-sample and held-out-reference metrics at the same
                # checkpoint so later plots can compare them directly.
                record = {
                    "step": step,
                    "rho": rho,
                    "f_mem": f_mem,
                    "baseline_f_mem": baseline_f_mem,
                    "delta_f_mem": f_mem - baseline_f_mem,
                    "mean_r": float(r_gen.mean().item()),
                    "median_r": float(r_gen.median().item()),
                    "baseline_mean_r": float(r_base.mean().item()),
                    "baseline_median_r": float(r_base.median().item()),
                    "mean_d1": float(generated_stats["d1"].mean().item()),
                    "mean_d2": float(generated_stats["d2"].mean().item()),
                    "baseline_mean_d1": float(baseline_stats["d1"].mean().item()),
                    "baseline_mean_d2": float(baseline_stats["d2"].mean().item()),
                }
                records.append(record)
                if not progress.disable:
                    progress.set_postfix(
                        f_mem=f"{record['f_mem']:.4f}",
                        r=f"{record['mean_r']:.4f}",
                        refresh=False,
                    )
                tracker.evaluation_progress(
                    checkpoint_index=checkpoint_index,
                    total_checkpoints=len(checkpoint_paths),
                    record=record,
                    elapsed_sec=time.perf_counter() - eval_start,
                    force=True,
                )

                if checkpoint_path == checkpoint_paths[-1]:
                    # Save the final samples and their nearest neighbors for later
                    # panels.
                    neighbor_stats = compute_nn_stats(
                        samples,
                        train_tensor,
                        chunk_size=int(config.evaluation.chunk_size),
                    )
                    panel_count = min(
                        int(config.evaluation.sample_panel_count),
                        samples.shape[0],
                    )
                    save_tensor(
                        run_dir / "metrics" / "final_samples.pt",
                        samples[:panel_count],
                    )
                    save_tensor(
                        run_dir / "metrics" / "final_neighbors.pt",
                        train_tensor[neighbor_stats["nn_idx"][:panel_count]],
                    )
        finally:
            progress.close()

        save_json(run_dir / "metrics" / "checkpoint_metrics.json", records)
        save_json(run_dir / "metrics" / "loss_eval.json", loss_records)
        save_json(
            run_dir / "metrics" / "checkpoint_sample_previews.json",
            checkpoint_sample_previews,
        )
        figure_paths = _save_checkpoint_metric_figures(records, run_dir / "figures")
        for name, path in figure_paths.items():
            tracker.register_figure(name, path)

        run_summary_path = run_dir / "artifacts" / "run_summary.json"
        run_summary = load_json(run_summary_path) if run_summary_path.exists() else {}
        evaluation_wall_time_sec = time.perf_counter() - eval_start
        final_record = records[-1]
        final_loss_record = loss_records[-1] if loss_records else {}
        run_summary.update(
            {
                "evaluation_wall_time_sec": evaluation_wall_time_sec,
                "checkpoint_metrics_path": str(
                    (run_dir / "metrics" / "checkpoint_metrics.json").resolve()
                ),
                "final_samples_path": str((run_dir / "metrics" / "final_samples.pt").resolve()),
                "final_neighbors_path": str(
                    (run_dir / "metrics" / "final_neighbors.pt").resolve()
                ),
                "loss_eval_path": str((run_dir / "metrics" / "loss_eval.json").resolve()),
                "checkpoint_sample_previews_path": str(
                    (run_dir / "metrics" / "checkpoint_sample_previews.json").resolve()
                ),
                "checkpoint_sample_previews": checkpoint_sample_previews,
                "final_metrics": final_record,
                "final_loss_eval": final_loss_record,
                "generated_figures": tracker.status.get("generated_figures"),
                "wandb": tracker.status.get("wandb"),
            }
        )
        save_json(run_summary_path, run_summary)
        tracker.log_metrics(
            {
                "eval/final_f_mem": final_record["f_mem"],
                "eval/final_baseline_f_mem": final_record["baseline_f_mem"],
                "eval/final_delta_f_mem": final_record["delta_f_mem"],
                "eval/final_mean_r": final_record["mean_r"],
                "eval/final_median_r": final_record["median_r"],
                "eval/final_baseline_mean_r": final_record["baseline_mean_r"],
                "eval/final_baseline_median_r": final_record["baseline_median_r"],
                "eval/evaluation_wall_time_sec": evaluation_wall_time_sec,
                "eval/final_random_t_loss": final_loss_record.get("random_t_loss"),
                "eval/final_fixed_t_loss_mean": final_loss_record.get("fixed_t_loss_mean"),
            },
            step=int(final_record["step"]),
        )
        tracker.stage(
            "completed",
            message=(
                "checkpoint evaluation complete "
                f"checkpoints={len(records)} "
                f"wall_time={evaluation_wall_time_sec:.2f}s"
            ),
            checkpoint_metrics_path=str(
                (run_dir / "metrics" / "checkpoint_metrics.json").resolve()
            ),
            loss_eval_path=str((run_dir / "metrics" / "loss_eval.json").resolve()),
            checkpoint_sample_previews_path=str(
                (run_dir / "metrics" / "checkpoint_sample_previews.json").resolve()
            ),
            checkpoint_sample_previews=checkpoint_sample_previews,
            final_samples_path=str((run_dir / "metrics" / "final_samples.pt").resolve()),
            final_neighbors_path=str(
                (run_dir / "metrics" / "final_neighbors.pt").resolve()
            ),
            generated_figures=tracker.status.get("generated_figures"),
        )
        return records
    except KeyboardInterrupt as exc:
        if tracker is not None:
            tracker.stage("aborted", last_error=f"{type(exc).__name__}: {exc}")
        else:
            update_attempt_status(run_dir, "aborted", last_error=f"{type(exc).__name__}: {exc}")
        raise
    except Exception as exc:
        if tracker is not None:
            tracker.stage("failed", last_error=f"{type(exc).__name__}: {exc}")
        else:
            update_attempt_status(run_dir, "failed", last_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    print("mp_runners.evaluate_checkpoints example")

    # `evaluate_run(...)` works on an already-finished run directory rather than
    # on in-memory tensors. The example below shows the expected call pattern.
    print("Usage:")
    print("  from mp_runners.evaluate_checkpoints import evaluate_run")
    print("  records = evaluate_run(run_dir, device='cpu')")
    print("")

    # These artifacts are the evaluation inputs. They fix the data tensors and
    # shared `z_init` so checkpoint comparisons use the same starting point.
    print("Expected artifacts under run_dir/artifacts:")
    print("  - config.json")
    print("  - train_tensor.pt")
    print("  - reference_tensor.pt")
    print("  - z_init.pt")
    print("")

    # These checkpoints are the states that will be evaluated and scored.
    print("Expected checkpoints under run_dir/checkpoints:")
    print("  - step_000000.pt")
    print("  - step_000050.pt")
    print("  - ...")
