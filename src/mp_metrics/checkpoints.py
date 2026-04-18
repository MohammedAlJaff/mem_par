"""
Checkpoint evaluation for first-party study runs.

Reload a saved run, regenerate samples at each checkpoint, and compute the
memorization curves.
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
from mp_metrics.memorization import compute_memorization_metrics, compute_nn_stats
from mp_tracking import AttemptTracker


def _validate_saved_holdout_metadata(run_dir):
    """
    Re-check saved train/reference source metadata before checkpoint evaluation.
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

    nn_ratio_path = figures_dir / "nn_ratio_vs_tau.png"
    means = [record["mean_nn_ratio"] for record in records]
    stds = [record["std_nn_ratio"] for record in records]
    baseline_means = [record["baseline_mean_nn_ratio"] for record in records]
    baseline_stds = [record["baseline_std_nn_ratio"] for record in records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, means, marker="o", label="generated mean")
    ax.fill_between(
        steps,
        [mean - std for mean, std in zip(means, stds)],
        [mean + std for mean, std in zip(means, stds)],
        alpha=0.18,
        label="generated +/- std",
    )
    ax.plot(steps, baseline_means, marker="o", linestyle="--", label="held-out mean")
    ax.fill_between(
        steps,
        [mean - std for mean, std in zip(baseline_means, baseline_stds)],
        [mean + std for mean, std in zip(baseline_means, baseline_stds)],
        alpha=0.14,
        label="held-out +/- std",
    )
    ax.set_title("Nearest-neighbor ratio over training")
    ax.set_xlabel("checkpoint step tau")
    ax.set_ylabel("mean nn ratio")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(nn_ratio_path, dpi=160)
    plt.close(fig)
    figure_paths["nn_ratio_vs_tau"] = nn_ratio_path

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
    payload so later plotting code can show concrete examples instead of just
    scalar curves.
    """
    run_dir = Path(run_dir)
    tracker = None
    try:
        config = load_config_snapshot(run_dir)
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
                model.load_state_dict(state["model"])
                samples = model.sample(
                    z_init.clone().to(device),
                    n_steps=int(config.sampling.n_steps),
                    method=str(config.sampling.method),
                    device=device,
                ).cpu()

                metrics = compute_memorization_metrics(
                    samples,
                    train_tensor,
                    chunk_size=int(config.evaluation.chunk_size),
                    config=config,
                )
                baseline = compute_memorization_metrics(
                    reference_tensor[: samples.shape[0]],
                    train_tensor,
                    chunk_size=int(config.evaluation.chunk_size),
                    config=config,
                )

                # Keep both generated-sample and held-out-reference metrics at the same
                # checkpoint so later plots can compare them directly.
                record = {
                    "step": int(state["step"]),
                    "ratio_threshold": metrics["ratio_threshold"],
                    "f_mem": metrics["f_mem"],
                    "mean_nn_ratio": metrics["mean_nn_ratio"],
                    "std_nn_ratio": metrics["std_nn_ratio"],
                    "mean_d1": metrics["mean_d1"],
                    "mean_d2": metrics["mean_d2"],
                    "baseline_f_mem": baseline["f_mem"],
                    "baseline_mean_nn_ratio": baseline["mean_nn_ratio"],
                    "baseline_std_nn_ratio": baseline["std_nn_ratio"],
                }
                records.append(record)
                if not progress.disable:
                    progress.set_postfix(
                        f_mem=f"{record['f_mem']:.4f}",
                        nn_ratio=f"{record['mean_nn_ratio']:.4f}",
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
                        train_tensor[neighbor_stats["indices"][:panel_count]],
                    )
        finally:
            progress.close()

        save_json(run_dir / "metrics" / "checkpoint_metrics.json", records)
        figure_paths = _save_checkpoint_metric_figures(records, run_dir / "figures")
        for name, path in figure_paths.items():
            tracker.register_figure(name, path)

        run_summary_path = run_dir / "artifacts" / "run_summary.json"
        run_summary = load_json(run_summary_path) if run_summary_path.exists() else {}
        evaluation_wall_time_sec = time.perf_counter() - eval_start
        final_record = records[-1]
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
                "final_metrics": final_record,
                "generated_figures": tracker.status.get("generated_figures"),
                "wandb": tracker.status.get("wandb"),
            }
        )
        save_json(run_summary_path, run_summary)
        tracker.log_metrics(
            {
                "eval/final_f_mem": final_record["f_mem"],
                "eval/final_mean_nn_ratio": final_record["mean_nn_ratio"],
                "eval/final_std_nn_ratio": final_record["std_nn_ratio"],
                "eval/final_baseline_f_mem": final_record["baseline_f_mem"],
                "eval/final_baseline_mean_nn_ratio": final_record["baseline_mean_nn_ratio"],
                "eval/final_baseline_std_nn_ratio": final_record["baseline_std_nn_ratio"],
                "eval/evaluation_wall_time_sec": evaluation_wall_time_sec,
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
    print("mp_metrics.checkpoints example")

    # `evaluate_run(...)` works on an already-finished run directory rather than
    # on in-memory tensors. The example below shows the expected call pattern.
    print("Usage:")
    print("  from mp_metrics import evaluate_run")
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
