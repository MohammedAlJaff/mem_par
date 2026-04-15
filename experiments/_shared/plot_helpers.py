from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from mp_artifacts import load_config_snapshot, load_json


def _restore_image_shape(tensor, config):
    if tensor.dim() == 4:
        return tensor

    if int(config.data.image_size) > 0 and int(config.data.image_channels) > 0:
        return tensor.view(
            tensor.shape[0],
            int(config.data.image_channels),
            int(config.data.image_size),
            int(config.data.image_size),
        )
    return tensor


def _plot_history(history, evaluation, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    if history:
        axes[0].plot([row["step"] for row in history], [row["loss"] for row in history])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)

    if evaluation:
        steps = [row["step"] for row in evaluation]
        axes[1].plot(steps, [row["f_mem"] for row in evaluation], label="f_mem")
        axes[1].plot(
            steps,
            [row["mean_nn_ratio"] for row in evaluation],
            label="mean_nn_ratio",
        )
        axes[1].plot(
            steps,
            [row["baseline_mean_nn_ratio"] for row in evaluation],
            label="baseline_mean_nn_ratio",
            linestyle="--",
        )
        axes[1].legend()
    axes[1].set_title("Checkpoint Metrics")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("metric value")
    axes[1].grid(alpha=0.3)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_panel(samples, neighbors, config, output_path):
    samples = _restore_image_shape(samples, config)
    neighbors = _restore_image_shape(neighbors, config)
    count = min(samples.shape[0], neighbors.shape[0])

    if samples.dim() == 4:
        fig, axes = plt.subplots(2, count, figsize=(2.2 * count, 4.5), constrained_layout=True)
        if count == 1:
            axes = axes.reshape(2, 1)
        for index in range(count):
            sample = samples[index]
            neighbor = neighbors[index]

            if sample.shape[0] == 1:
                axes[0, index].imshow(sample[0], cmap="gray", vmin=-1.0, vmax=1.0)
                axes[1, index].imshow(neighbor[0], cmap="gray", vmin=-1.0, vmax=1.0)
            else:
                axes[0, index].imshow(((sample.permute(1, 2, 0) + 1.0) / 2.0).clamp(0.0, 1.0))
                axes[1, index].imshow(((neighbor.permute(1, 2, 0) + 1.0) / 2.0).clamp(0.0, 1.0))

            axes[0, index].set_title(f"sample {index}")
            axes[1, index].set_title(f"nearest {index}")
            axes[0, index].axis("off")
            axes[1, index].axis("off")
    else:
        fig, axes = plt.subplots(count, 2, figsize=(10, 2.4 * count), constrained_layout=True)
        if count == 1:
            axes = [axes]
        for index in range(count):
            row = axes[index]
            row[0].plot(samples[index].detach().cpu())
            row[1].plot(neighbors[index].detach().cpu(), color="tab:orange")
            row[0].set_title(f"sample {index}")
            row[1].set_title(f"nearest {index}")
            row[0].grid(alpha=0.3)
            row[1].grid(alpha=0.3)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_run(run_dir, study_dir=None):
    run_dir = Path(run_dir)
    if study_dir is None:
        study_dir = run_dir.parent.parent
    study_dir = Path(study_dir)
    figures_dir = study_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_snapshot(run_dir)
    history = load_json(run_dir / "metrics" / "train_history.json")
    evaluation = load_json(run_dir / "metrics" / "checkpoint_metrics.json")

    run_name = run_dir.name
    _plot_history(history, evaluation, figures_dir / f"{run_name}_metrics.png")

    final_samples_path = run_dir / "metrics" / "final_samples.pt"
    final_neighbors_path = run_dir / "metrics" / "final_neighbors.pt"
    if final_samples_path.exists() and final_neighbors_path.exists():
        final_samples = torch.load(final_samples_path, map_location="cpu")
        final_neighbors = torch.load(final_neighbors_path, map_location="cpu")
        _plot_panel(
            final_samples,
            final_neighbors,
            config,
            figures_dir / f"{run_name}_panel.png",
        )

    return figures_dir
