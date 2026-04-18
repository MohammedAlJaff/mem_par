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
from mp_metrics.memorization import compute_nn_stats


DEFAULT_FIXED_TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)


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


def collect_loss_eval_record(
    step,
    model,
    train_tensor,
    config,
    device,
    *,
    fixed_times=DEFAULT_FIXED_TIMES,
):
    """Evaluate loss once with training-style random t and at fixed t probes."""
    batch_size = min(int(config.evaluation.batch_size), int(train_tensor.shape[0]))
    batch = train_tensor[:batch_size].to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            torch.manual_seed(int(config.training.seed) + int(step) + 1009)
            random_t_loss = float(model.compute_loss(batch).detach().cpu().item())

            fixed = {}
            for value in fixed_times:
                t_value = float(value)
                torch.manual_seed(int(config.training.seed) + 2027)
                eps = torch.randn_like(batch)
                t = torch.full((batch.shape[0],), t_value, device=batch.device)
                loss = model.compute_loss(batch, eps=eps, t=t)
                fixed[_time_key(t_value)] = float(loss.detach().cpu().item())
    finally:
        model.train(was_training)

    fixed_values = list(fixed.values())
    record = {
        "step": int(step),
        "random_t_loss": random_t_loss,
        "fixed_t_loss_mean": float(sum(fixed_values) / max(1, len(fixed_values))),
        "fixed_t": fixed,
    }
    for key, value in fixed.items():
        record[f"loss_t_{key}"] = value
    return record


def evaluate_image32_run(run_dir, device="cpu", fixed_times=DEFAULT_FIXED_TIMES):
    """Evaluate checkpoint memorization and fixed/random-t losses."""
    run_dir = Path(run_dir)
    config = load_config_snapshot(run_dir)
    train_tensor = torch.load(run_dir / "artifacts" / "train_tensor.pt", map_location="cpu")
    reference_tensor = torch.load(
        run_dir / "artifacts" / "reference_tensor.pt",
        map_location="cpu",
    )
    z_init = torch.load(run_dir / "artifacts" / "z_init.pt", map_location="cpu")

    model = GDFM(config).to(device)
    records = []
    loss_records = []
    checkpoint_paths = list_checkpoint_paths(run_dir)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

    for checkpoint_path in checkpoint_paths:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        samples = model.sample(
            z_init.clone().to(device),
            n_steps=int(config.sampling.n_steps),
            method=str(config.sampling.method),
            device=device,
        ).cpu()

        records.append(
            collect_image32_metric_record(
                int(state["step"]),
                samples,
                reference_tensor,
                train_tensor,
                chunk_size=int(config.evaluation.chunk_size),
                config=config,
            )
        )
        loss_records.append(
            collect_loss_eval_record(
                int(state["step"]),
                model,
                train_tensor,
                config,
                device,
                fixed_times=fixed_times,
            )
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
    _update_run_summary(run_dir, records, loss_records)
    return {
        "checkpoint_metrics": records,
        "loss_eval": loss_records,
    }


def _update_run_summary(run_dir, records, loss_records):
    run_dir = Path(run_dir)
    run_summary_path = run_dir / "artifacts" / "run_summary.json"
    run_summary = load_json(run_summary_path) if run_summary_path.exists() else {}
    run_summary.update(
        {
            "checkpoint_metrics_path": str(
                (run_dir / "metrics" / "checkpoint_metrics.json").resolve()
            ),
            "loss_eval_path": str((run_dir / "metrics" / "loss_eval.json").resolve()),
            "final_samples_path": str((run_dir / "metrics" / "final_samples.pt").resolve()),
            "final_neighbors_path": str((run_dir / "metrics" / "final_neighbors.pt").resolve()),
            "final_metrics": records[-1] if records else {},
            "final_loss_eval": loss_records[-1] if loss_records else {},
        }
    )
    save_json(run_summary_path, run_summary)


def _time_key(value):
    return str(float(value)).replace(".", "_")


__all__ = [
    "DEFAULT_FIXED_TIMES",
    "collect_image32_metric_record",
    "collect_loss_eval_record",
    "collect_reference_memory",
    "evaluate_image32_run",
]
