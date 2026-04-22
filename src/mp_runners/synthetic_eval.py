"""
Synthetic checkpoint evaluation for saved synthetic-runner attempts.

This module handles saved-run replay, tensor preview artifacts, synthetic
figures, run-summary updates, and tracker logging. Metric definitions stay in
`mp_metrics.synthetic`.
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
from mp_metrics.loss_eval import collect_loss_eval_record
from mp_metrics.synthetic import (
    collect_controlled_synthetic_metric_record,
    intrinsic_geometry_parts,
)
from mp_runtime import autocast_context, configure_torch_runtime
from mp_tracking import AttemptTracker, loss_eval_metric_payload
from mp_visualization import make_synthetic_figures


def _sliced_directions(config, intrinsic_dim):
    """
    Draw fixed random directions for intrinsic sliced-W2.

    The seed offset keeps these metric directions separate from data sampling
    while making checkpoint replay deterministic.
    """
    count = int(getattr(config, "sliced_w2_projections", 128))
    generator = torch.Generator().manual_seed(int(config.training.seed) + 1009)
    directions = torch.randn(count, int(intrinsic_dim), generator=generator)
    return directions / directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)


def synthetic_metric_payload(record):
    """Return tracker metric keys for one synthetic metric record."""
    payload = {
        "eval/ambient_nearest_neighbor_ratio_d1_over_d2": record.get(
            "ambient_nearest_neighbor_ratio_d1_over_d2"
        ),
        "eval/intrinsic_nearest_neighbor_ratio_d1_over_d2": record.get(
            "intrinsic_nearest_neighbor_ratio_d1_over_d2"
        ),
        "eval/intrinsic_sliced_w2": record.get("intrinsic_sliced_w2"),
        "eval/intrinsic_gmm_nll_mean": record.get("intrinsic_gmm_nll_mean"),
        "eval/holdout_ambient_nearest_neighbor_ratio_d1_over_d2": record.get(
            "holdout_ambient_nearest_neighbor_ratio_d1_over_d2"
        ),
        "eval/holdout_intrinsic_nearest_neighbor_ratio_d1_over_d2": record.get(
            "holdout_intrinsic_nearest_neighbor_ratio_d1_over_d2"
        ),
        "eval/holdout_intrinsic_sliced_w2": record.get("holdout_intrinsic_sliced_w2"),
        "eval/holdout_intrinsic_gmm_nll_mean": record.get(
            "holdout_intrinsic_gmm_nll_mean"
        ),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _figure_context(config):
    """Build the context text shown under synthetic figures."""
    dataset = str(config.data.dataset)
    if dataset == "torus3d":
        dimension_line = (
            f"{dataset} D={int(config.data.ambient_dim)} "
            "base=3 local=2 angle_features=4 "
            f"N={int(config.data.n_train)} ref={int(config.data.n_eval)}"
        )
    else:
        dimension_line = (
            f"{dataset} D={int(config.data.ambient_dim)} "
            f"d={int(config.data.intrinsic_data_dim)} "
            f"N={int(config.data.n_train)} ref={int(config.data.n_eval)}"
        )
    return (
        f"{dimension_line}\n"
        f"MLP layers={int(config.network.n_layers)} hidden={int(config.network.hidden_dim)} "
        f"time_emb={int(config.network.time_emb_dim)} "
        f"pred={config.framework.prediction_space} loss={config.framework.loss_space} "
        f"sampler={config.sampling.method}x{int(config.sampling.n_steps)}\n"
        f"seed={int(config.training.seed)} steps={int(config.training.n_steps)} "
        f"batch={int(config.training.batch_cap)} lr={float(config.training.lr):.3g} "
        f"wd={float(config.training.weight_decay):.3g}"
    )


def _centers_for_figures(config, geometry):
    """Return GMM centers for synthetic figure overlays."""
    if str(config.data.dataset) != "gmm_subspace":
        return None, None
    intrinsic_centers = geometry["means"]
    ambient_centers = intrinsic_centers @ geometry["basis"].T
    return intrinsic_centers, ambient_centers


def evaluate_synthetic_run(run_dir, device="cpu", fixed_times=None):
    """
    Replay controlled-synthetic evaluation for one saved attempt directory.

    The evaluator loads saved train/reference tensors, saved geometry, and the
    saved `z_init`. Every checkpoint is sampled from that same `z_init`.
    """
    run_dir = Path(run_dir)
    config = load_config_snapshot(run_dir)
    configure_torch_runtime(config, device)
    tracker = None
    try:
        tracker = AttemptTracker(config, run_dir)
        tracker.stage("synthetic_evaluating", message=f"Synthetic evaluation started device={device}")

        train_tensor = torch.load(run_dir / "artifacts" / "train_tensor.pt", map_location="cpu")
        reference_tensor = torch.load(
            run_dir / "artifacts" / "reference_tensor.pt",
            map_location="cpu",
        )
        z_init = torch.load(run_dir / "artifacts" / "z_init.pt", map_location="cpu")
        geometry = torch.load(run_dir / "artifacts" / "geometry.pt", map_location="cpu")

        # Intrinsic coordinates are computed once from saved tensors and reused
        # by every checkpoint. This keeps ambient and intrinsic metrics aligned.
        train_intrinsic = intrinsic_geometry_parts(train_tensor, geometry, config=config)["intrinsic"]
        reference_intrinsic = intrinsic_geometry_parts(
            reference_tensor,
            geometry,
            config=config,
        )["intrinsic"]
        intrinsic_dim = int(train_intrinsic.shape[1])
        sliced_directions = _sliced_directions(config, intrinsic_dim)
        holdout_count = min(int(config.evaluation.n_samples), int(reference_tensor.shape[0]))
        holdout_tensor = reference_tensor[:holdout_count]
        holdout_intrinsic = reference_intrinsic[:holdout_count]

        model = GDFM(config).to(device)
        records = []
        loss_records = []
        checkpoint_sample_previews = []
        checkpoint_paths = list_checkpoint_paths(run_dir)
        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

        final_samples = None
        for checkpoint_path in checkpoint_paths:
            # Use the same saved initial noise at every checkpoint for fair replay.
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

            record = collect_controlled_synthetic_metric_record(
                step=step,
                samples=samples,
                baseline_samples=holdout_tensor,
                train_tensor=train_tensor,
                train_intrinsic=train_intrinsic,
                geometry=geometry,
                reference_intrinsic=holdout_intrinsic,
                sliced_directions=sliced_directions,
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
                    **synthetic_metric_payload(record),
                    **loss_eval_metric_payload(loss_record),
                },
                step=step,
            )

            preview_record = _collect_tensor_preview_record(
                step=step,
                samples=samples,
                train_tensor=train_tensor,
                run_dir=run_dir,
                config=config,
                figure_enabled=False,
            )
            checkpoint_sample_previews.append(preview_record)

            if checkpoint_path == checkpoint_paths[-1]:
                final_samples = samples

        save_json(run_dir / "metrics" / "checkpoint_metrics.json", records)
        save_json(run_dir / "metrics" / "metrics_along_training.json", records)
        save_json(run_dir / "metrics" / "loss_eval.json", loss_records)
        save_json(
            run_dir / "metrics" / "checkpoint_sample_previews.json",
            checkpoint_sample_previews,
        )

        final_parts = intrinsic_geometry_parts(final_samples, geometry, config=config)
        save_tensor(run_dir / "metrics" / "final_samples.pt", final_samples)
        save_tensor(run_dir / "metrics" / "final_samples_ambient.pt", final_samples)
        save_tensor(
            run_dir / "metrics" / "final_samples_intrinsic.pt",
            final_parts["intrinsic"],
        )
        if str(config.data.dataset) == "gmm_subspace":
            save_tensor(run_dir / "metrics" / "final_samples_projected.pt", final_parts["projected"])
        if str(config.data.dataset) == "torus3d":
            save_tensor(run_dir / "metrics" / "final_samples_torus_xyz.pt", final_parts["torus_xyz"])
            save_tensor(run_dir / "metrics" / "final_samples_angle_features.pt", final_parts["intrinsic"])

        train_history_path = run_dir / "metrics" / "train_history.json"
        train_history = load_json(train_history_path) if train_history_path.exists() else []
        intrinsic_centers, ambient_centers = _centers_for_figures(config, geometry)
        figure_paths = make_synthetic_figures(
            figures_dir=run_dir / "figures",
            train_history=train_history,
            metric_records=records,
            train_tensor=train_tensor,
            reference_tensor=reference_tensor,
            final_samples=final_samples,
            train_intrinsic=train_intrinsic,
            reference_intrinsic=reference_intrinsic,
            final_samples_intrinsic=final_parts["intrinsic"],
            geometry=geometry,
            intrinsic_centers=intrinsic_centers,
            ambient_centers=ambient_centers,
            context=_figure_context(config),
        )
        for name, path in figure_paths.items():
            tracker.register_figure(name, path)

        run_summary_path = run_dir / "artifacts" / "run_summary.json"
        run_summary = load_json(run_summary_path) if run_summary_path.exists() else {}
        final_record = records[-1] if records else {}
        final_loss_record = loss_records[-1] if loss_records else {}
        run_summary.update(
            {
                "checkpoint_metrics_path": str(
                    (run_dir / "metrics" / "checkpoint_metrics.json").resolve()
                ),
                "metrics_along_training_path": str(
                    (run_dir / "metrics" / "metrics_along_training.json").resolve()
                ),
                "loss_eval_path": str((run_dir / "metrics" / "loss_eval.json").resolve()),
                "checkpoint_sample_previews_path": str(
                    (run_dir / "metrics" / "checkpoint_sample_previews.json").resolve()
                ),
                "checkpoint_sample_previews": checkpoint_sample_previews,
                "final_samples_path": str((run_dir / "metrics" / "final_samples.pt").resolve()),
                "final_samples_intrinsic_path": str(
                    (run_dir / "metrics" / "final_samples_intrinsic.pt").resolve()
                ),
                "final_metrics": final_record,
                "final_loss_eval": final_loss_record,
                "figures": figure_paths,
            }
        )
        save_json(run_summary_path, run_summary)

        tracker.log_metrics(
            {
                "eval/final_ambient_nearest_neighbor_ratio_d1_over_d2": final_record.get(
                    "ambient_nearest_neighbor_ratio_d1_over_d2"
                ),
                "eval/final_intrinsic_nearest_neighbor_ratio_d1_over_d2": final_record.get(
                    "intrinsic_nearest_neighbor_ratio_d1_over_d2"
                ),
                "eval/final_intrinsic_sliced_w2": final_record.get("intrinsic_sliced_w2"),
                "eval/final_intrinsic_gmm_nll_mean": final_record.get(
                    "intrinsic_gmm_nll_mean"
                ),
                "eval/final_holdout_ambient_nearest_neighbor_ratio_d1_over_d2": final_record.get(
                    "holdout_ambient_nearest_neighbor_ratio_d1_over_d2"
                ),
                "eval/final_holdout_intrinsic_nearest_neighbor_ratio_d1_over_d2": final_record.get(
                    "holdout_intrinsic_nearest_neighbor_ratio_d1_over_d2"
                ),
                "eval/final_holdout_intrinsic_sliced_w2": final_record.get(
                    "holdout_intrinsic_sliced_w2"
                ),
                "eval/final_holdout_intrinsic_gmm_nll_mean": final_record.get(
                    "holdout_intrinsic_gmm_nll_mean"
                ),
                "eval/final_random_t_loss": final_loss_record.get("random_t_loss"),
                "eval/final_fixed_t_loss_mean": final_loss_record.get("fixed_t_loss_mean"),
            },
            step=int(final_record["step"]),
        )
        tracker.stage(
            "completed",
            message=f"Synthetic evaluation complete checkpoints={len(records)}",
            checkpoint_sample_previews_path=str(
                (run_dir / "metrics" / "checkpoint_sample_previews.json").resolve()
            ),
            checkpoint_sample_previews=checkpoint_sample_previews,
            figures=figure_paths,
        )
        return {
            "checkpoint_metrics": records,
            "loss_eval": loss_records,
            "checkpoint_sample_previews": checkpoint_sample_previews,
            "figures": figure_paths,
            "final_metrics": final_record,
            "final_loss_eval": final_loss_record,
        }
    except Exception as exc:
        if tracker is not None:
            tracker.stage("failed", last_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if tracker is not None:
            tracker.close()


def _preview_count(config, samples, train_tensor=None, count=None):
    """Resolve how many checkpoint samples to save."""
    if count is None:
        evaluation = getattr(config, "evaluation", None) if config is not None else None
        count = int(getattr(evaluation, "sample_panel_count", 8))
    limit = int(samples.shape[0])
    if train_tensor is not None:
        limit = min(limit, int(train_tensor.shape[0]))
    return max(0, min(int(count), limit))


def _collect_tensor_preview_record(
    *,
    step,
    samples,
    train_tensor,
    run_dir,
    config=None,
    count=None,
    figure_enabled=False,
):
    """
    Save checkpoint sample tensors for synthetic evaluation.

    Synthetic preview records store tensors rather than image panels. The main
    visual evidence is produced later by `make_synthetic_figures`.
    """
    del train_tensor, figure_enabled
    step = int(step)
    run_dir = Path(run_dir)
    count = _preview_count(config, samples, count=count)
    samples_dir = run_dir / "metrics" / "checkpoint_sample_previews"
    samples_path = samples_dir / f"step_{step:06d}.pt"
    samples_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples[:count].detach().cpu(), samples_path)
    return {
        "step": step,
        "count": int(count),
        "samples_path": str(samples_path.resolve()),
        "figure_path": None,
    }


__all__ = [
    "evaluate_synthetic_run",
    "synthetic_metric_payload",
]


if __name__ == "__main__":
    print("mp_runners.synthetic_eval example")
    print("Use evaluate_synthetic_run(run_dir) on a saved DONE attempt.")
