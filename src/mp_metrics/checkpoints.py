from pathlib import Path

import torch

from gdfm import GDFM
from mp_artifacts import list_checkpoint_paths, load_config_snapshot, save_json, save_tensor
from mp_metrics.memorization import compute_memorization_metrics, nearest_neighbor_stats


def evaluate_run(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    config = load_config_snapshot(run_dir)

    train_tensor = torch.load(run_dir / "artifacts" / "train_tensor.pt", map_location="cpu")
    reference_tensor = torch.load(
        run_dir / "artifacts" / "reference_tensor.pt", map_location="cpu"
    )
    z_init = torch.load(run_dir / "artifacts" / "z_init.pt", map_location="cpu")

    model = GDFM(config).to(device)
    records = []
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

        metrics = compute_memorization_metrics(
            samples,
            train_tensor,
            ratio_threshold=float(config.evaluation.ratio_threshold),
            chunk_size=int(config.evaluation.chunk_size),
        )
        baseline = compute_memorization_metrics(
            reference_tensor[: samples.shape[0]],
            train_tensor,
            ratio_threshold=float(config.evaluation.ratio_threshold),
            chunk_size=int(config.evaluation.chunk_size),
        )
        record = {
            "step": int(state["step"]),
            "f_mem": metrics["f_mem"],
            "mean_nn_ratio": metrics["mean_nn_ratio"],
            "mean_d1": metrics["mean_d1"],
            "mean_d2": metrics["mean_d2"],
            "baseline_f_mem": baseline["f_mem"],
            "baseline_mean_nn_ratio": baseline["mean_nn_ratio"],
        }
        records.append(record)

        if checkpoint_path == checkpoint_paths[-1]:
            neighbor_stats = nearest_neighbor_stats(
                samples,
                train_tensor,
                chunk_size=int(config.evaluation.chunk_size),
            )
            panel_count = min(int(config.evaluation.sample_panel_count), samples.shape[0])
            save_tensor(
                run_dir / "metrics" / "final_samples.pt",
                samples[:panel_count],
            )
            save_tensor(
                run_dir / "metrics" / "final_neighbors.pt",
                train_tensor[neighbor_stats["indices"][:panel_count]],
            )

    save_json(run_dir / "metrics" / "checkpoint_metrics.json", records)
    return records
