"""
Simple spiral memorization demo for flow matching.

This script does four things in one run:
    1. Sample N noiseless points from a 2D spiral and save the dataset.
    2. Train a tiny time-conditioned flow-matching MLP on those points.
    3. Every k updates, sample from the model and measure memorization with
       the nearest-neighbor ratio r = d1 / d2.
    4. Save checkpoints, rich plots, and GIF animations of training dynamics.

Outputs are written under:
    output_dir/
        config.json
        summary.json
        data/
        checkpoints/
        figures/
        animations/
        samples/

Example:
    python -m experiments.synthetic.low_d.spiral_mem_demo.run --device cpu --n 64 --train-steps 2000
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Iterable, List

TRAIN_COLOR = "#4c78a8"
GEN_COLOR = "#e45756"
SOURCE_COLOR = "#6c757d"


def default_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_runtime_dependencies() -> None:
    global np
    global plt
    global torch
    global DataLoader
    global TensorDataset
    global compute_baseline_nn_stats
    global compute_f_mem
    global compute_nn_stats
    global FlowMatching
    global TimeConditionedMLP
    global CosineWarmup
    global save_metrics
    global seed_everything

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_module
    import numpy as np_module
    import torch as torch_module
    from torch.utils.data import DataLoader as DataLoaderClass
    from torch.utils.data import TensorDataset as TensorDatasetClass

    from experiments.synthetic.low_d.common.evaluation.memorization import (
        compute_baseline_nn_stats as compute_baseline_nn_stats_fn,
        compute_f_mem as compute_f_mem_fn,
        compute_nn_stats as compute_nn_stats_fn,
    )
    from experiments.synthetic.low_d.common.flow_matching import FlowMatching as FlowMatchingCls
    from experiments.synthetic.low_d.common.networks.mlp import TimeConditionedMLP as TimeConditionedMLPCls
    from experiments.synthetic.low_d.common.training.lr_schedule import (
        CosineWarmup as CosineWarmupCls,
    )
    from experiments.synthetic.low_d.common.utils import (
        save_metrics as save_metrics_fn,
        seed_everything as seed_everything_fn,
    )

    np = np_module
    plt = plt_module
    torch = torch_module
    DataLoader = DataLoaderClass
    TensorDataset = TensorDatasetClass
    compute_baseline_nn_stats = compute_baseline_nn_stats_fn
    compute_f_mem = compute_f_mem_fn
    compute_nn_stats = compute_nn_stats_fn
    FlowMatching = FlowMatchingCls
    TimeConditionedMLP = TimeConditionedMLPCls
    CosineWarmup = CosineWarmupCls
    save_metrics = save_metrics_fn
    seed_everything = seed_everything_fn


def log(message: str, start_time: float | None = None) -> None:
    prefix = "[spiral-demo]"
    if start_time is not None:
        prefix = f"{prefix}[{time.time() - start_time:7.1f}s]"
    print(f"{prefix} {message}", flush=True)


def configure_visual_style() -> None:
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook")
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "grid.color": "#b0b8c2",
            "axes.facecolor": "#fbfbfc",
            "figure.facecolor": "white",
        }
    )


def apply_grid(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.28, linewidth=0.8, color="#b0b8c2")
    ax.set_axisbelow(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny flow-matching model on a noiseless spiral and track memorization."
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of training points on the spiral."
    )
    parser.add_argument(
        "--turns", type=float, default=3.0, help="Number of spiral turns."
    )
    parser.add_argument(
        "--radius-min", type=float, default=0.15, help="Spiral radius at the start."
    )
    parser.add_argument(
        "--radius-max", type=float, default=2.25, help="Spiral radius at the end."
    )
    parser.add_argument(
        "--x-offset",
        type=float,
        default=3.0,
        help="Shift the spiral this many units to the right in raw coordinates.",
    )
    parser.add_argument(
        "--normalize-data",
        action="store_true",
        help="Center/scale the spiral before training. Disabled by default so the model trains on the raw shifted coordinates.",
    )
    parser.add_argument(
        "--train-steps", type=int, default=20000, help="Number of gradient updates."
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Legacy option. Evaluation scheduling now uses --num-evals.",
    )
    parser.add_argument(
        "--num-evals",
        type=int,
        default=10,
        help="Number of evaluation snapshots, spaced evenly from step 0 to the final training step.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Training batch size."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay."
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.05,
        help="Fraction of training steps used for linear LR warmup.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden width.")
    parser.add_argument("--depth", type=int, default=4, help="MLP depth.")
    parser.add_argument(
        "--time-embed-dim", type=int, default=64, help="Time embedding width."
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "silu"],
        default="silu",
        help="MLP activation.",
    )
    parser.add_argument(
        "--pred-space",
        choices=["x", "eps", "v"],
        default="v",
        help="What the network predicts.",
    )
    parser.add_argument(
        "--loss-space",
        choices=["x", "eps", "v"],
        default="v",
        help="Where the loss is measured.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=50,
        help="Number of ODE integration steps used for generation.",
    )
    parser.add_argument(
        "--sampler",
        choices=["euler", "heun"],
        default="heun",
        help="ODE sampler used during evaluation snapshots.",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=4000,
        help="Generated samples per evaluation snapshot.",
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=1600,
        help="How many trajectories to show in the trajectory plots/animations.",
    )
    parser.add_argument(
        "--holdout-multiplier",
        type=int,
        default=8,
        help="Holdout size multiplier relative to N.",
    )
    parser.add_argument(
        "--reference-points",
        type=int,
        default=4096,
        help="Dense spiral reference size for quality metrics and plotting.",
    )
    parser.add_argument(
        "--mmd-subsample",
        type=int,
        default=512,
        help="Subsample size for MMD/Chamfer evaluation.",
    )
    parser.add_argument(
        "--mem-threshold",
        type=float,
        default=0.5,
        help="Threshold for f_mem based on the mem-ratio d1/d2.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument(
        "--device",
        default=default_device(),
        help="cpu or cuda.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Directory where all artifacts are saved.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=180,
        help="Raster DPI for saved figures.",
    )
    parser.add_argument(
        "--skip-gifs",
        action="store_true",
        help="Skip GIF assembly and only keep the PNG frames.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print a training progress line every this many gradient steps.",
    )
    return parser.parse_args()


def make_dirs(root: Path) -> dict[str, Path]:
    """
    Create directories for the spiral memorization demo.

    Args:
        root: Root directory for the run.

    Returns:
        Dictionary of created directories.
    """
    dirs = {
        "root": root,
        "data": root / "data",
        "checkpoints": root / "checkpoints",
        "figures": root / "figures",
        "snapshots": root / "figures" / "snapshots",
        "anim_frames": root / "figures" / "animation_frames",
        "sampling_frames": root / "figures" / "sampling_frames",
        "animations": root / "animations",
        "samples": root / "samples",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def make_unique_run_root(base_dir: Path, args: argparse.Namespace) -> Path:
    """
    Create a unique run root directory.

    Args:
        base_dir: Base directory for the run.
        args: Command-line arguments.

    Returns:
        Unique run root directory.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stem = (
        f"{timestamp}_n{args.n}_steps{args.train_steps}_"
        f"pred{args.pred_space}_loss{args.loss_space}_"
        f"sampler{args.sampler}_{'norm' if args.normalize_data else 'raw'}_"
        f"xoff{args.x_offset:g}_seed{args.seed}"
    )
    candidate = base_dir / stem
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base_dir / f"{stem}_v{suffix}"
    return candidate


def sample_spiral_from_parameter(
    s: torch.Tensor,
    turns: float,
    radius_min: float,
    radius_max: float,
    x_offset: float = 0.0,
) -> torch.Tensor:
    """
    Sample points from a spiral parameterized by s.

    Args:
        s: Parameter values in [0, 1].
        turns: Number of spiral turns.
        radius_min: Minimum radius.
        radius_max: Maximum radius.
        x_offset: Offset applied to the x-coordinate.

    Returns:
        Tensor of shape (N, 2) with (x, y) coordinates.
    """
    theta = s * (2.0 * math.pi * turns)
    radius = radius_min + (radius_max - radius_min) * s
    x = radius * torch.cos(theta) + x_offset
    y = radius * torch.sin(theta)
    return torch.stack([x, y], dim=1)


def build_spiral_sets(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    train_s = torch.linspace(0.0, 1.0, args.n)
    holdout_n = max(args.n * args.holdout_multiplier, args.num_eval_samples)
    holdout_s = (torch.arange(holdout_n, dtype=torch.float32) + 0.5) / holdout_n
    holdout_s = holdout_s.clamp(max=1.0)
    reference_s = torch.linspace(0.0, 1.0, args.reference_points)

    train_raw = sample_spiral_from_parameter(
        train_s, args.turns, args.radius_min, args.radius_max, x_offset=args.x_offset
    )
    holdout_raw = sample_spiral_from_parameter(
        holdout_s, args.turns, args.radius_min, args.radius_max, x_offset=args.x_offset
    )
    reference_raw = sample_spiral_from_parameter(
        reference_s,
        args.turns,
        args.radius_min,
        args.radius_max,
        x_offset=args.x_offset,
    )

    mean = train_raw.mean(dim=0)
    std = train_raw.std(dim=0).clamp(min=1e-6)

    return {
        "train_s": train_s,
        "holdout_s": holdout_s,
        "reference_s": reference_s,
        "train_raw": train_raw,
        "holdout_raw": holdout_raw,
        "reference_raw": reference_raw,
        "train_model": normalize_points(
            train_raw, mean, std, enabled=args.normalize_data
        ),
        "holdout_model": normalize_points(
            holdout_raw, mean, std, enabled=args.normalize_data
        ),
        "reference_model": normalize_points(
            reference_raw, mean, std, enabled=args.normalize_data
        ),
        "normalize_data": bool(args.normalize_data),
        "mean": mean,
        "std": std,
    }


def normalize_points(
    points: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    enabled: bool = True,
) -> torch.Tensor:
    """
    Normalize points using mean and standard deviation.

    Args:
        points: Points to normalize.
        mean: Mean values.
        std: Standard deviation values.
        enabled: Whether to normalize.

    Returns:
        Normalized points.
    """
    if not enabled:
        return points.clone()
    return (points - mean) / std


def denormalize_points(
    points: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    enabled: bool = True,
) -> torch.Tensor:
    if not enabled:
        return points.clone()
    return points * std + mean


def denormalize_trajectory(
    trajectory: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    enabled: bool = True,
) -> torch.Tensor:
    if not enabled:
        return trajectory.clone()
    return trajectory * std.view(1, 1, -1) + mean.view(1, 1, -1)


def cycle_loader(loader: DataLoader) -> Iterable[torch.Tensor]:
    while True:
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                yield batch[0]
            else:
                yield batch


def sample_with_trajectory(
    flow_model: FlowMatching,
    initial_noise: torch.Tensor,
    n_steps: int,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        z = initial_noise.clone()
        dt = 1.0 / n_steps
        t_values = torch.linspace(0.0, 1.0, n_steps + 1, device=z.device)
        trajectory = [z.detach().cpu()]

        flow_model.eval()
        for i in range(n_steps):
            t_curr = float(t_values[i].item())
            t_next = float(t_values[i + 1].item())
            t_batch = torch.full((z.shape[0],), t_curr, device=z.device).clamp(1e-3, 0.999)
            v1 = flow_model._get_velocity(z, t_batch)
            if method == "euler":
                z = z + dt * v1
            else:
                if method != "heun":
                    raise ValueError(f"Unknown sampler method: {method}")
                t_next_batch = torch.full((z.shape[0],), t_next, device=z.device).clamp(
                    1e-3, 0.999
                )
                z_predict = z + dt * v1
                if i < n_steps - 1:
                    v2 = flow_model._get_velocity(z_predict, t_next_batch)
                    z = z + 0.5 * dt * (v1 + v2)
                else:
                    z = z_predict
            trajectory.append(z.detach().cpu())

        return z.detach().cpu(), torch.stack(trajectory, dim=0)


def make_reference_subset(reference_model: torch.Tensor, size: int) -> torch.Tensor:
    if reference_model.shape[0] <= size:
        return reference_model.clone()
    idx = torch.linspace(0, reference_model.shape[0] - 1, size).long()
    return reference_model[idx]


def pairwise_median_distance(points: torch.Tensor) -> float:
    if points.shape[0] < 2:
        return 1.0
    dists = torch.pdist(points)
    positive = dists[dists > 0]
    if positive.numel() == 0:
        return 1.0
    return float(torch.median(positive).item())


def rbf_kernel_matrix(
    x: torch.Tensor, y: torch.Tensor, sigmas: List[float]
) -> torch.Tensor:
    sq_dists = torch.cdist(x, y, p=2.0) ** 2
    kernel = torch.zeros_like(sq_dists)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma * sigma)
        kernel = kernel + torch.exp(-gamma * sq_dists)
    return kernel / len(sigmas)


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas: List[float]) -> float:
    k_xx = rbf_kernel_matrix(x, x, sigmas)
    k_yy = rbf_kernel_matrix(y, y, sigmas)
    k_xy = rbf_kernel_matrix(x, y, sigmas)

    n = x.shape[0]
    m = y.shape[0]

    xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / max(n * (n - 1), 1)
    yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / max(m * (m - 1), 1)
    xy = k_xy.mean()
    mmd2 = xx + yy - 2.0 * xy
    return float(torch.sqrt(torch.clamp(mmd2, min=0.0)).item())


def compute_chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    d_xy = torch.cdist(x, y).min(dim=1).values.mean()
    d_yx = torch.cdist(y, x).min(dim=1).values.mean()
    return float(0.5 * (d_xy + d_yx).item())


def moving_average(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.size < window:
        window = arr.size
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_dataset_artifacts(
    dirs: dict[str, Path], data: dict[str, torch.Tensor], args: argparse.Namespace
) -> None:
    np.savez(
        dirs["data"] / "spiral_dataset.npz",
        train_s=data["train_s"].numpy(),
        holdout_s=data["holdout_s"].numpy(),
        reference_s=data["reference_s"].numpy(),
        train_raw=data["train_raw"].numpy(),
        holdout_raw=data["holdout_raw"].numpy(),
        reference_raw=data["reference_raw"].numpy(),
        train_model=data["train_model"].numpy(),
        holdout_model=data["holdout_model"].numpy(),
        reference_model=data["reference_model"].numpy(),
        mean=data["mean"].numpy(),
        std=data["std"].numpy(),
        normalize_data=np.array(data["normalize_data"], dtype=np.bool_),
    )
    torch.save(data, dirs["data"] / "spiral_dataset.pt")
    save_metrics(
        {
            "n": args.n,
            "turns": args.turns,
            "radius_min": args.radius_min,
            "radius_max": args.radius_max,
            "x_offset": args.x_offset,
            "normalize_data": bool(args.normalize_data),
            "holdout_size": int(data["holdout_raw"].shape[0]),
            "reference_size": int(data["reference_raw"].shape[0]),
        },
        str(dirs["data"] / "dataset_info.json"),
    )


def plot_dataset_overview(
    data: dict[str, torch.Tensor],
    output_path: Path,
    dpi: int,
) -> None:
    train_raw = data["train_raw"].numpy()
    holdout_raw = data["holdout_raw"].numpy()
    reference_raw = data["reference_raw"].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        reference_raw[:, 0],
        reference_raw[:, 1],
        color="#c7c7c7",
        linewidth=1.0,
        label="dense reference",
    )
    axes[0].scatter(
        holdout_raw[:, 0],
        holdout_raw[:, 1],
        s=4,
        color=TRAIN_COLOR,
        alpha=0.12,
        label="holdout",
    )
    axes[0].scatter(
        train_raw[:, 0],
        train_raw[:, 1],
        s=10,
        color=TRAIN_COLOR,
        alpha=0.75,
        label="train",
    )
    axes[0].set_title("Spiral Dataset")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="best")
    axes[0].set_aspect("equal", adjustable="box")
    apply_grid(axes[0])

    axes[1].plot(
        reference_raw[:, 0], reference_raw[:, 1], color="#c7c7c7", linewidth=1.0
    )
    axes[1].scatter(
        train_raw[:, 0],
        train_raw[:, 1],
        s=10,
        color=TRAIN_COLOR,
        alpha=0.75,
    )
    axes[1].set_title("Training Points Ordered Along the Spiral")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal", adjustable="box")
    apply_grid(axes[1])

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def compute_plot_bounds(
    *point_sets: torch.Tensor, padding_fraction: float = 0.12, min_padding: float = 0.5
) -> tuple[float, float, float, float]:
    if not point_sets:
        raise ValueError("compute_plot_bounds requires at least one point set.")

    processed = []
    for points in point_sets:
        if points is None or points.numel() == 0:
            continue
        processed.append(points.detach().cpu())

    if not processed:
        raise ValueError("compute_plot_bounds received only empty point sets.")

    mins = (
        torch.stack([points.min(dim=0).values for points in processed], dim=0)
        .min(dim=0)
        .values
    )
    maxs = (
        torch.stack([points.max(dim=0).values for points in processed], dim=0)
        .max(dim=0)
        .values
    )
    span = (maxs - mins).max().item()
    padding = max(min_padding, padding_fraction * span)
    return (
        float(mins[0].item() - padding),
        float(maxs[0].item() + padding),
        float(mins[1].item() - padding),
        float(maxs[1].item() + padding),
    )


def compute_velocity_grid(
    flow_model: FlowMatching,
    mean: torch.Tensor,
    std: torch.Tensor,
    normalize_data: bool,
    bounds: tuple[float, float, float, float],
    device: torch.device,
    t_value: float = 0.5,
    grid_size: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        x_min, x_max, y_min, y_max = bounds
        x_vals = np.linspace(x_min, x_max, grid_size)
        y_vals = np.linspace(y_min, y_max, grid_size)
        gx, gy = np.meshgrid(x_vals, y_vals)
        grid_raw = torch.tensor(
            np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1),
            dtype=torch.float32,
            device=device,
        )
        grid_model = normalize_points(
            grid_raw, mean.to(device), std.to(device), enabled=normalize_data
        )
        t_batch = torch.full(
            (grid_model.shape[0],), t_value, dtype=torch.float32, device=device
        )
        velocity_model = flow_model._get_velocity(grid_model, t_batch)
        velocity_raw = denormalize_points(
            velocity_model,
            torch.zeros_like(mean).to(device),
            std.to(device),
            enabled=normalize_data,
        )
        vx = velocity_raw[:, 0].cpu().numpy().reshape(grid_size, grid_size)
        vy = velocity_raw[:, 1].cpu().numpy().reshape(grid_size, grid_size)
        return gx, gy, vx, vy


def plot_snapshot_panel(
    output_path: Path,
    data: dict[str, torch.Tensor],
    generated_raw: torch.Tensor,
    trajectory_raw: torch.Tensor,
    nn_stats: dict,
    baseline_stats: dict,
    snapshot_row: dict,
    eval_history: list[dict],
    flow_model: FlowMatching,
    bounds: tuple[float, float, float, float],
    device: torch.device,
    num_traj_samples: int,
    dpi: int,
) -> None:
    train_raw = data["train_raw"].numpy()
    holdout_raw = data["holdout_raw"].numpy()
    reference_raw = data["reference_raw"].numpy()
    generated_np = generated_raw.numpy()
    ratio_np = nn_stats["r"].numpy()
    baseline_ratio_np = baseline_stats["r"].numpy()
    mem_mask = ratio_np < snapshot_row["mem_threshold"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].plot(
        reference_raw[:, 0],
        reference_raw[:, 1],
        color="#cfcfcf",
        linewidth=1.0,
        label="dense reference",
    )
    axes[0, 0].scatter(
        holdout_raw[:, 0],
        holdout_raw[:, 1],
        s=4,
        color=TRAIN_COLOR,
        alpha=0.08,
        label="holdout",
    )
    axes[0, 0].scatter(
        train_raw[:, 0],
        train_raw[:, 1],
        s=10,
        color=TRAIN_COLOR,
        alpha=0.75,
        label="train",
    )
    axes[0, 0].scatter(
        generated_np[:, 0],
        generated_np[:, 1],
        s=4,
        color=GEN_COLOR,
        alpha=0.12,
        label="generated",
    )
    axes[0, 0].set_title(f"Generated Samples at Step {snapshot_row['step']}")
    axes[0, 0].set_xlim(bounds[0], bounds[1])
    axes[0, 0].set_ylim(bounds[2], bounds[3])
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].legend(loc="best")
    axes[0, 0].set_aspect("equal", adjustable="box")
    apply_grid(axes[0, 0])

    axes[0, 1].plot(
        reference_raw[:, 0], reference_raw[:, 1], color="#dddddd", linewidth=1.0
    )
    axes[0, 1].scatter(
        generated_np[:, 0],
        generated_np[:, 1],
        s=4,
        color=GEN_COLOR,
        alpha=0.12,
        label="generated",
    )
    if mem_mask.any():
        axes[0, 1].scatter(
            generated_np[mem_mask, 0],
            generated_np[mem_mask, 1],
            facecolors="none",
            edgecolors=GEN_COLOR,
            linewidths=0.35,
            s=10,
            label="r < threshold",
        )
    axes[0, 1].scatter(
        train_raw[:, 0],
        train_raw[:, 1],
        s=10,
        color=TRAIN_COLOR,
        alpha=0.75,
        label="train",
    )
    axes[0, 1].set_title("Train vs Generated Density")
    axes[0, 1].set_xlim(bounds[0], bounds[1])
    axes[0, 1].set_ylim(bounds[2], bounds[3])
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_aspect("equal", adjustable="box")
    axes[0, 1].legend(loc="best")
    apply_grid(axes[0, 1])

    axes[0, 2].hist(
        baseline_ratio_np,
        bins=30,
        density=True,
        alpha=0.45,
        color=TRAIN_COLOR,
        label="holdout vs train",
    )
    axes[0, 2].hist(
        ratio_np,
        bins=30,
        density=True,
        alpha=0.55,
        color=GEN_COLOR,
        label="generated vs train",
    )
    axes[0, 2].axvline(
        snapshot_row["mem_threshold"], color="#7f7f7f", linestyle="--", linewidth=1.0
    )
    axes[0, 2].set_title("Mem-Ratio Distribution")
    axes[0, 2].set_xlabel("mem-ratio = d1 / d2")
    axes[0, 2].set_ylabel("density")
    axes[0, 2].legend(loc="best")
    apply_grid(axes[0, 2])
    axes[0, 2].text(
        0.98,
        0.98,
        (
            f"mean mem-ratio = {snapshot_row['mean_mem_ratio']:.3f}\n"
            f"f_mem = {snapshot_row['f_mem']:.3f}\n"
            f"baseline f_mem = {snapshot_row['baseline_f_mem']:.3f}"
        ),
        transform=axes[0, 2].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    gx, gy, vx, vy = compute_velocity_grid(
        flow_model=flow_model,
        mean=data["mean"],
        std=data["std"],
        normalize_data=data["normalize_data"],
        bounds=bounds,
        device=device,
    )
    axes[1, 0].plot(
        reference_raw[:, 0], reference_raw[:, 1], color="#dddddd", linewidth=1.0
    )
    axes[1, 0].quiver(gx, gy, vx, vy, color="#2a6f97", alpha=0.85)
    axes[1, 0].scatter(
        train_raw[:, 0], train_raw[:, 1], s=10, color=TRAIN_COLOR, alpha=0.75
    )
    axes[1, 0].set_title("Predicted Velocity Field at t=0.5")
    axes[1, 0].set_xlim(bounds[0], bounds[1])
    axes[1, 0].set_ylim(bounds[2], bounds[3])
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[1, 0].set_aspect("equal", adjustable="box")
    apply_grid(axes[1, 0])

    trajectory_subset = trajectory_raw[
        :, : min(trajectory_raw.shape[1], num_traj_samples), :
    ].numpy()
    axes[1, 1].plot(
        reference_raw[:, 0], reference_raw[:, 1], color="#dddddd", linewidth=1.0
    )
    axes[1, 1].scatter(
        trajectory_subset[0, :, 0],
        trajectory_subset[0, :, 1],
        s=4,
        color=SOURCE_COLOR,
        alpha=0.14,
        label="source noise",
    )
    for idx in range(trajectory_subset.shape[1]):
        axes[1, 1].plot(
            trajectory_subset[:, idx, 0],
            trajectory_subset[:, idx, 1],
            color=GEN_COLOR,
            alpha=0.025,
            linewidth=0.35,
        )
        axes[1, 1].scatter(
            trajectory_subset[0, idx, 0],
            trajectory_subset[0, idx, 1],
            color=SOURCE_COLOR,
            s=3,
            alpha=0.12,
        )
        axes[1, 1].scatter(
            trajectory_subset[-1, idx, 0],
            trajectory_subset[-1, idx, 1],
            color=GEN_COLOR,
            s=3,
            alpha=0.12,
        )
    axes[1, 1].scatter(
        train_raw[:, 0], train_raw[:, 1], s=10, color=TRAIN_COLOR, alpha=0.75
    )
    axes[1, 1].set_title("Sample Paths Through ODE Integration")
    axes[1, 1].set_xlim(bounds[0], bounds[1])
    axes[1, 1].set_ylim(bounds[2], bounds[3])
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].legend(loc="best")
    apply_grid(axes[1, 1])

    steps = np.array([row["step"] for row in eval_history], dtype=np.float64)
    f_mem_values = np.array([row["f_mem"] for row in eval_history], dtype=np.float64)
    mmd_values = np.array([row["mmd_rbf"] for row in eval_history], dtype=np.float64)
    axes[1, 2].plot(steps, f_mem_values, marker="o", color="#d62728", label="f_mem")
    axes[1, 2].axhline(
        snapshot_row["baseline_f_mem"],
        color="#4c78a8",
        linestyle="--",
        label="baseline f_mem",
    )
    axes[1, 2].set_xlabel("gradient updates")
    axes[1, 2].set_ylabel("f_mem", color="#d62728")
    axes[1, 2].tick_params(axis="y", labelcolor="#d62728")
    axes[1, 2].set_title("Memorization and Quality Over Training")
    ax2 = axes[1, 2].twinx()
    ax2.plot(steps, mmd_values, marker="s", color="#2a9d8f", label="MMD")
    ax2.set_ylabel("MMD", color="#2a9d8f")
    ax2.tick_params(axis="y", labelcolor="#2a9d8f")

    lines_a, labels_a = axes[1, 2].get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    axes[1, 2].legend(lines_a + lines_b, labels_a + labels_b, loc="best")
    apply_grid(axes[1, 2])

    fig.suptitle(
        (
            f"Spiral Flow Matching Snapshot | step={snapshot_row['step']} | "
            f"N={snapshot_row['train_size']} | pred={snapshot_row['pred_space']} | loss={snapshot_row['loss_space']}"
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_simple_animation_frame(
    output_path: Path,
    data: dict[str, torch.Tensor],
    generated_raw: torch.Tensor,
    source_raw: torch.Tensor,
    nn_stats: dict,
    snapshot_row: dict,
    bounds: tuple[float, float, float, float],
    dpi: int,
) -> None:
    train_raw = data["train_raw"].numpy()
    reference_raw = data["reference_raw"].numpy()
    generated_np = generated_raw.numpy()
    source_np = source_raw.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        reference_raw[:, 0], reference_raw[:, 1], color="#d9d9d9", linewidth=1.0
    )
    axes[0].scatter(
        source_np[:, 0],
        source_np[:, 1],
        s=4,
        color=SOURCE_COLOR,
        alpha=0.35,
        label="source noise",
    )
    axes[0].scatter(
        train_raw[:, 0],
        train_raw[:, 1],
        s=10,
        color=TRAIN_COLOR,
        alpha=0.75,
        label="train",
    )
    axes[0].scatter(
        generated_np[:, 0],
        generated_np[:, 1],
        s=4,
        color=GEN_COLOR,
        alpha=0.12,
        label="generated",
    )
    axes[0].set_xlim(bounds[0], bounds[1])
    axes[0].set_ylim(bounds[2], bounds[3])
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"Generated Samples | step {snapshot_row['step']}")
    axes[0].legend(loc="best")
    apply_grid(axes[0])

    text = (
        f"train size: {snapshot_row['train_size']}\n"
        f"mean mem-ratio: {snapshot_row['mean_mem_ratio']:.3f}\n"
        f"f_mem: {snapshot_row['f_mem']:.3f}\n"
        f"MMD: {snapshot_row['mmd_rbf']:.3f}\n"
        f"Chamfer: {snapshot_row['chamfer']:.3f}\n"
        f"loss: {snapshot_row['loss']:.4f}"
    )
    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.95,
        text,
        fontsize=16,
        va="top",
        ha="left",
        family="monospace",
    )
    axes[1].set_title("Snapshot Metrics")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_metric_summary(
    loss_history: list[dict],
    eval_history: list[dict],
    output_path: Path,
    dpi: int,
) -> None:
    loss_steps = [row["step"] for row in loss_history]
    losses = [row["loss"] for row in loss_history]
    smoothed_loss = moving_average(
        losses, window=min(50, max(5, len(losses) // 20 or 1))
    )

    eval_steps = [row["step"] for row in eval_history]
    mean_ratio = [row["mean_mem_ratio"] for row in eval_history]
    f_mem = [row["f_mem"] for row in eval_history]
    mmd = [row["mmd_rbf"] for row in eval_history]
    chamfer = [row["chamfer"] for row in eval_history]
    d1 = [row["mean_d1"] for row in eval_history]
    d2 = [row["mean_d2"] for row in eval_history]
    baseline_f_mem = eval_history[0]["baseline_f_mem"]
    baseline_ratio = eval_history[0]["baseline_mean_mem_ratio"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].plot(loss_steps, losses, color="#adb5bd", alpha=0.55, label="raw loss")
    axes[0, 0].plot(
        loss_steps, smoothed_loss, color="#1d3557", linewidth=2.0, label="smoothed loss"
    )
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("gradient updates")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend(loc="best")
    apply_grid(axes[0, 0])

    axes[0, 1].plot(eval_steps, mean_ratio, marker="o", color="#6a4c93")
    axes[0, 1].axhline(
        baseline_ratio, color="#4c78a8", linestyle="--", label="holdout baseline"
    )
    axes[0, 1].set_title("Mean Mem-Ratio vs Gradient Updates")
    axes[0, 1].set_xlabel("gradient updates")
    axes[0, 1].set_ylabel("mean mem-ratio = mean(d1/d2)")
    axes[0, 1].legend(loc="best")
    apply_grid(axes[0, 1])

    axes[0, 2].plot(eval_steps, f_mem, marker="o", color="#d62728")
    axes[0, 2].axhline(
        baseline_f_mem, color="#4c78a8", linestyle="--", label="holdout baseline"
    )
    axes[0, 2].set_title("Memorization Fraction vs Gradient Updates")
    axes[0, 2].set_xlabel("gradient updates")
    axes[0, 2].set_ylabel("f_mem")
    axes[0, 2].legend(loc="best")
    apply_grid(axes[0, 2])

    axes[1, 0].plot(eval_steps, mmd, marker="s", color="#2a9d8f")
    axes[1, 0].set_title("RBF-MMD to Dense Spiral Reference")
    axes[1, 0].set_xlabel("gradient updates")
    axes[1, 0].set_ylabel("MMD")
    apply_grid(axes[1, 0])

    axes[1, 1].plot(eval_steps, chamfer, marker="s", color="#e9c46a")
    axes[1, 1].set_title("Chamfer Distance to Dense Spiral")
    axes[1, 1].set_xlabel("gradient updates")
    axes[1, 1].set_ylabel("Chamfer")
    apply_grid(axes[1, 1])

    axes[1, 2].plot(eval_steps, d1, marker="o", color="#ff7f0e", label="mean d1")
    axes[1, 2].plot(eval_steps, d2, marker="^", color="#1f77b4", label="mean d2")
    axes[1, 2].set_title("Nearest-Neighbor Distances")
    axes[1, 2].set_xlabel("gradient updates")
    axes[1, 2].set_ylabel("distance")
    axes[1, 2].legend(loc="best")
    apply_grid(axes[1, 2])

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_single_metric(
    eval_history: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
    baseline_key: str | None = None,
) -> None:
    steps = [row["step"] for row in eval_history]
    values = [row[metric_key] for row in eval_history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, values, marker="o", linewidth=2.0, color="#1d3557")
    if baseline_key is not None:
        ax.axhline(
            eval_history[0][baseline_key],
            linestyle="--",
            color="#4c78a8",
            label="holdout baseline",
        )
        ax.legend(loc="best")
    ax.set_xlabel("gradient updates")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    apply_grid(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_sampling_trajectory_frames(
    trajectory_raw: torch.Tensor,
    data: dict[str, torch.Tensor],
    bounds: tuple[float, float, float, float],
    out_dir: Path,
    num_traj_samples: int,
    dpi: int,
) -> list[Path]:
    reference_raw = data["reference_raw"].numpy()
    train_raw = data["train_raw"].numpy()
    subset = trajectory_raw[
        :, : min(trajectory_raw.shape[1], num_traj_samples), :
    ].numpy()
    frame_paths = []

    for frame_idx in range(1, subset.shape[0]):
        frame_path = out_dir / f"sampling_{frame_idx:04d}.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(
            reference_raw[:, 0], reference_raw[:, 1], color="#d9d9d9", linewidth=1.0
        )
        ax.scatter(
            train_raw[:, 0],
            train_raw[:, 1],
            s=14,
            color=TRAIN_COLOR,
            alpha=0.75,
            label="train",
        )

        for sample_idx in range(subset.shape[1]):
            ax.plot(
                subset[1 : frame_idx + 1, sample_idx, 0],
                subset[1 : frame_idx + 1, sample_idx, 1],
                color=GEN_COLOR,
                linewidth=0.35,
                alpha=0.02,
            )
        ax.scatter(
            subset[frame_idx, :, 0],
            subset[frame_idx, :, 1],
            s=12,
            color=GEN_COLOR,
            alpha=0.35,
            label="current particles",
        )
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"Final Model Sampling Trajectories | ODE step {frame_idx}/{subset.shape[0] - 1}"
        )
        ax.legend(loc="best")
        apply_grid(ax)
        fig.tight_layout()
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        frame_paths.append(frame_path)

    return frame_paths


def make_gif(
    frame_paths: list[Path], output_path: Path, duration_ms: int = 250
) -> bool:
    if not frame_paths:
        return False
    try:
        from PIL import Image
    except Exception:
        return False

    images = [Image.open(path) for path in frame_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=1,
    )
    for image in images:
        image.close()
    return True


def snapshot_step_set(train_steps: int, num_evals: int) -> list[int]:
    if num_evals <= 1:
        return [0, train_steps] if train_steps > 0 else [0]
    snapshot_steps = np.linspace(0, train_steps, num=num_evals)
    snapshot_steps = np.rint(snapshot_steps).astype(int)
    return sorted(set(snapshot_steps.tolist()))


def evaluate_snapshot(
    step: int,
    current_loss: float,
    flow_model: FlowMatching,
    optimizer: torch.optim.Optimizer,
    data: dict[str, torch.Tensor],
    dirs: dict[str, Path],
    args: argparse.Namespace,
    baseline_stats: dict,
    metric_reference_model: torch.Tensor,
    metric_sigmas: list[float],
    eval_noise: torch.Tensor,
    bounds: tuple[float, float, float, float],
    device: torch.device,
    eval_history: list[dict],
) -> tuple[dict, torch.Tensor]:
    generated_model, trajectory_model = sample_with_trajectory(
        flow_model=flow_model,
        initial_noise=eval_noise,
        n_steps=args.sample_steps,
        method=args.sampler,
    )
    generated_raw = denormalize_points(
        generated_model,
        data["mean"],
        data["std"],
        enabled=data["normalize_data"],
    )
    trajectory_raw = denormalize_trajectory(
        trajectory_model,
        data["mean"],
        data["std"],
        enabled=data["normalize_data"],
    )

    nn_stats = compute_nn_stats(generated_raw, data["train_raw"])
    f_mem = compute_f_mem(nn_stats["r"], threshold=args.mem_threshold)
    mean_ratio = float(nn_stats["r"].mean().item())
    median_ratio = float(nn_stats["r"].median().item())
    mean_d1 = float(nn_stats["d1"].mean().item())
    mean_d2 = float(nn_stats["d2"].mean().item())

    metric_count = min(
        args.mmd_subsample, generated_model.shape[0], metric_reference_model.shape[0]
    )
    gen_metric_model = generated_model[:metric_count]
    ref_metric_model = metric_reference_model[:metric_count]
    mmd_rbf = compute_mmd_rbf(gen_metric_model, ref_metric_model, metric_sigmas)
    chamfer = compute_chamfer_distance(gen_metric_model, ref_metric_model)

    snapshot_row = {
        "step": int(step),
        "loss": float(current_loss),
        "train_size": int(data["train_raw"].shape[0]),
        "pred_space": args.pred_space,
        "loss_space": args.loss_space,
        "mem_threshold": float(args.mem_threshold),
        "mean_mem_ratio": mean_ratio,
        "median_mem_ratio": median_ratio,
        "mean_nn_ratio": mean_ratio,
        "median_nn_ratio": median_ratio,
        "f_mem": float(f_mem),
        "mean_d1": mean_d1,
        "mean_d2": mean_d2,
        "mmd_rbf": float(mmd_rbf),
        "chamfer": float(chamfer),
        "baseline_mean_mem_ratio": float(baseline_stats["r"].mean().item()),
        "baseline_median_mem_ratio": float(baseline_stats["r"].median().item()),
        "baseline_mean_ratio": float(baseline_stats["r"].mean().item()),
        "baseline_median_ratio": float(baseline_stats["r"].median().item()),
        "baseline_f_mem": float(
            compute_f_mem(baseline_stats["r"], threshold=args.mem_threshold)
        ),
        "baseline_mean_d1": float(baseline_stats["d1"].mean().item()),
        "baseline_mean_d2": float(baseline_stats["d2"].mean().item()),
    }
    eval_history.append(snapshot_row)

    torch.save(
        {
            "step": step,
            "model_state_dict": flow_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": snapshot_row,
        },
        dirs["checkpoints"] / f"step_{step:07d}.pt",
    )

    np.savez(
        dirs["samples"] / f"samples_step_{step:07d}.npz",
        generated=generated_raw.numpy(),
        generated_model=generated_model.numpy(),
        mem_ratio=nn_stats["r"].numpy(),
        nn_ratio=nn_stats["r"].numpy(),
        d1=nn_stats["d1"].numpy(),
        d2=nn_stats["d2"].numpy(),
        nn_idx=nn_stats["nn_idx"].numpy(),
        trajectory=trajectory_raw.numpy(),
        trajectory_model=trajectory_model.numpy(),
    )

    save_metrics(snapshot_row, str(dirs["samples"] / f"metrics_step_{step:07d}.json"))

    plot_snapshot_panel(
        output_path=dirs["snapshots"] / f"snapshot_{step:07d}.png",
        data=data,
        generated_raw=generated_raw,
        trajectory_raw=trajectory_raw,
        nn_stats=nn_stats,
        baseline_stats=baseline_stats,
        snapshot_row=snapshot_row,
        eval_history=eval_history,
        flow_model=flow_model,
        bounds=bounds,
        device=device,
        num_traj_samples=args.num_traj_samples,
        dpi=args.figure_dpi,
    )

    plot_simple_animation_frame(
        output_path=dirs["anim_frames"] / f"frame_{step:07d}.png",
        data=data,
        generated_raw=generated_raw,
        source_raw=trajectory_raw[0],
        nn_stats=nn_stats,
        snapshot_row=snapshot_row,
        bounds=bounds,
        dpi=args.figure_dpi,
    )

    return snapshot_row, trajectory_raw


def initial_loss_estimate(flow_model: FlowMatching, batch: torch.Tensor) -> float:
    flow_model.eval()
    with torch.no_grad():
        return float(flow_model.compute_loss(batch).item())


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    if args.n < 2:
        raise ValueError(
            "Need at least 2 training points to compute the first and second nearest neighbors."
        )
    if args.train_steps < 1:
        raise ValueError("--train-steps must be at least 1.")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be at least 1.")
    if args.sample_steps < 1:
        raise ValueError("--sample-steps must be at least 1.")
    if args.num_eval_samples < 1:
        raise ValueError("--num-eval-samples must be at least 1.")
    if args.num_traj_samples < 1:
        raise ValueError("--num-traj-samples must be at least 1.")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1.")
    if args.num_evals < 2:
        raise ValueError(
            "--num-evals must be at least 2 so we include the start and end of training."
        )

    start_time = time.time()
    log("Starting spiral memorization demo.", start_time)
    log(
        (
            f"Config: N={args.n}, train_steps={args.train_steps}, num_evals={args.num_evals}, "
            f"pred={args.pred_space}, loss={args.loss_space}, sampler={args.sampler}, "
            f"coords={'normalized' if args.normalize_data else 'raw'}, "
            f"x_offset={args.x_offset}, device={args.device}"
        ),
        start_time,
    )

    seed_everything(args.seed)
    log(f"Seeded all RNGs with seed={args.seed}.", start_time)
    configure_visual_style()
    log(
        "Configured seaborn-style plotting defaults with visible grid lines.",
        start_time,
    )

    base_root = Path(args.output_dir)
    root = make_unique_run_root(base_root, args)
    dirs = make_dirs(root)
    save_metrics(vars(args), str(root / "config.json"))
    log(f"Created output directories under {root} (base dir: {base_root}).", start_time)

    data = build_spiral_sets(args)
    log(
        (
            f"Built spiral datasets: train={data['train_raw'].shape[0]}, "
            f"holdout={data['holdout_raw'].shape[0]}, reference={data['reference_raw'].shape[0]}, "
            f"x_offset={args.x_offset}, normalize_data={args.normalize_data}"
        ),
        start_time,
    )
    if args.normalize_data:
        log(
            "Training will use centered/scaled coordinates; saved plots and artifacts remain in raw shifted coordinates.",
            start_time,
        )
    else:
        log(
            "Training will use the raw shifted coordinates directly, so the transport animation reflects the learned coordinate system.",
            start_time,
        )
    save_dataset_artifacts(dirs, data, args)
    plot_dataset_overview(
        data, dirs["figures"] / "dataset_overview.png", dpi=args.figure_dpi
    )
    log("Saved dataset artifacts and overview figure.", start_time)

    device = torch.device(args.device)
    train_dataset = TensorDataset(data["train_model"])
    loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    loader_iter = cycle_loader(loader)

    network = TimeConditionedMLP(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        time_embed_dim=args.time_embed_dim,
        activation=args.activation,
    )
    flow_model = FlowMatching(
        network=network,
        pred_space=args.pred_space,
        loss_space=args.loss_space,
    ).to(device)
    n_params = int(sum(parameter.numel() for parameter in flow_model.parameters()))
    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_steps = max(1, int(args.warmup_fraction * args.train_steps))
    scheduler = CosineWarmup(
        optimizer, total_steps=args.train_steps, warmup_steps=warmup_steps
    )
    log(
        (
            f"Initialized flow model with {n_params:,} parameters, AdamW optimizer "
            f"(lr={args.lr:.2e}), and cosine LR schedule "
            f"(warmup={warmup_steps} steps, {args.warmup_fraction:.0%} of training)."
        ),
        start_time,
    )

    baseline_stats = compute_baseline_nn_stats(data["holdout_raw"], data["train_raw"])
    metric_reference_model = make_reference_subset(
        data["reference_model"], args.mmd_subsample
    )
    sigma_base = pairwise_median_distance(metric_reference_model)
    metric_sigmas = [max(0.05, sigma_base * factor) for factor in (0.5, 1.0, 2.0)]
    baseline_f_mem = float(
        compute_f_mem(baseline_stats["r"], threshold=args.mem_threshold)
    )
    log(
        (
            f"Baseline holdout-vs-train stats: mean_mem_ratio={baseline_stats['r'].mean().item():.4f}, "
            f"f_mem={baseline_f_mem:.4f}"
        ),
        start_time,
    )

    eval_noise = torch.randn(args.num_eval_samples, 2, device=device)
    source_raw_for_bounds = denormalize_points(
        eval_noise.detach().cpu(),
        data["mean"],
        data["std"],
        enabled=data["normalize_data"],
    )
    bounds = compute_plot_bounds(data["reference_raw"], source_raw_for_bounds)
    log(
        (
            f"Using one fixed plotting frame covering both source and target: "
            f"x=[{bounds[0]:.2f}, {bounds[1]:.2f}], y=[{bounds[2]:.2f}, {bounds[3]:.2f}]"
        ),
        start_time,
    )
    loss_history: list[dict] = []
    eval_history: list[dict] = []
    final_trajectory_raw = None

    snapshot_steps = snapshot_step_set(args.train_steps, args.num_evals)
    preview_steps = snapshot_steps[:8]
    preview_suffix = " ..." if len(snapshot_steps) > 8 else ""
    log(
        f"Prepared {len(snapshot_steps)} evenly spaced evaluation snapshots at steps {preview_steps}{preview_suffix}.",
        start_time,
    )

    full_batch = data["train_model"].to(device)
    initial_loss = initial_loss_estimate(flow_model, full_batch)
    log(f"Initial full-batch loss estimate: {initial_loss:.6f}.", start_time)
    log(
        f"Running snapshot at step 0 with {args.sampler} sampling: generating model samples, computing memorization metrics, and saving plots.",
        start_time,
    )
    _, final_trajectory_raw = evaluate_snapshot(
        step=0,
        current_loss=initial_loss,
        flow_model=flow_model,
        optimizer=optimizer,
        data=data,
        dirs=dirs,
        args=args,
        baseline_stats=baseline_stats,
        metric_reference_model=metric_reference_model,
        metric_sigmas=metric_sigmas,
        eval_noise=eval_noise,
        bounds=bounds,
        device=device,
        eval_history=eval_history,
    )
    log(
        (
            f"Snapshot 0 saved: mean_mem_ratio={eval_history[-1]['mean_mem_ratio']:.4f}, "
            f"f_mem={eval_history[-1]['f_mem']:.4f}, MMD={eval_history[-1]['mmd_rbf']:.4f}"
        ),
        start_time,
    )

    for step in range(1, args.train_steps + 1):
        batch = next(loader_iter).to(device)
        flow_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = flow_model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.item())
        loss_history.append({"step": step, "loss": loss_value})

        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            recent_window = min(25, len(loss_history))
            recent_avg_loss = (
                sum(row["loss"] for row in loss_history[-recent_window:])
                / recent_window
            )
            current_lr = optimizer.param_groups[0]["lr"]
            log(
                (
                    f"Training step {step:6d}/{args.train_steps}: "
                    f"loss={loss_value:.6f}, recent_avg_loss={recent_avg_loss:.6f}, "
                    f"lr={current_lr:.2e}"
                ),
                start_time,
            )

        if step in snapshot_steps:
            log(
                (
                    f"Running snapshot at step {step} with {args.sampler} sampling: generating {args.num_eval_samples} samples, "
                    "computing mem-ratio memorization and quality metrics, and saving checkpoint/figures."
                ),
                start_time,
            )
            _, final_trajectory_raw = evaluate_snapshot(
                step=step,
                current_loss=loss_value,
                flow_model=flow_model,
                optimizer=optimizer,
                data=data,
                dirs=dirs,
                args=args,
                baseline_stats=baseline_stats,
                metric_reference_model=metric_reference_model,
                metric_sigmas=metric_sigmas,
                eval_noise=eval_noise,
                bounds=bounds,
                device=device,
                eval_history=eval_history,
            )
            log(
                (
                    f"Snapshot {step} saved: mean_mem_ratio={eval_history[-1]['mean_mem_ratio']:.4f}, "
                    f"f_mem={eval_history[-1]['f_mem']:.4f}, "
                    f"MMD={eval_history[-1]['mmd_rbf']:.4f}, "
                    f"Chamfer={eval_history[-1]['chamfer']:.4f}"
                ),
                start_time,
            )

    write_csv(loss_history, root / "loss_history.csv")
    write_csv(eval_history, root / "eval_history.csv")
    log("Wrote CSV histories for losses and evaluation snapshots.", start_time)

    log("Rendering summary figures.", start_time)
    plot_metric_summary(
        loss_history=loss_history,
        eval_history=eval_history,
        output_path=dirs["figures"] / "metric_summary.png",
        dpi=args.figure_dpi,
    )
    plot_single_metric(
        eval_history=eval_history,
        metric_key="mean_mem_ratio",
        ylabel="mean mem-ratio = mean(d1/d2)",
        title="Mean Mem-Ratio vs Gradient Updates",
        output_path=dirs["figures"] / "mean_nn_ratio_vs_updates.png",
        dpi=args.figure_dpi,
        baseline_key="baseline_mean_mem_ratio",
    )
    plot_single_metric(
        eval_history=eval_history,
        metric_key="f_mem",
        ylabel="f_mem",
        title="Memorization Fraction vs Gradient Updates",
        output_path=dirs["figures"] / "f_mem_vs_updates.png",
        dpi=args.figure_dpi,
        baseline_key="baseline_f_mem",
    )
    plot_single_metric(
        eval_history=eval_history,
        metric_key="mmd_rbf",
        ylabel="MMD",
        title="Generative Quality (RBF-MMD) vs Gradient Updates",
        output_path=dirs["figures"] / "mmd_vs_updates.png",
        dpi=args.figure_dpi,
    )
    log("Saved summary metric figures.", start_time)

    animation_frame_paths = sorted(dirs["anim_frames"].glob("frame_*.png"))
    snapshot_frame_paths = sorted(dirs["snapshots"].glob("snapshot_*.png"))

    if final_trajectory_raw is not None:
        log(
            "Rendering per-step sampling trajectory frames for the final model.",
            start_time,
        )
        sampling_frame_paths = plot_sampling_trajectory_frames(
            trajectory_raw=final_trajectory_raw,
            data=data,
            bounds=bounds,
            out_dir=dirs["sampling_frames"],
            num_traj_samples=args.num_traj_samples,
            dpi=args.figure_dpi,
        )
    else:
        sampling_frame_paths = []

    gif_status = {
        "training_samples_gif": False,
        "snapshot_panels_gif": False,
        "final_sampling_gif": False,
    }
    if not args.skip_gifs:
        log("Assembling GIF animations from saved PNG frames.", start_time)
        gif_status["training_samples_gif"] = make_gif(
            animation_frame_paths,
            dirs["animations"] / "training_samples.gif",
            duration_ms=180,
        )
        gif_status["snapshot_panels_gif"] = make_gif(
            snapshot_frame_paths,
            dirs["animations"] / "training_snapshot_panels.gif",
            duration_ms=180,
        )
        gif_status["final_sampling_gif"] = make_gif(
            sampling_frame_paths,
            dirs["animations"] / "final_sampling_paths.gif",
            duration_ms=30,
        )
        log(f"GIF assembly finished with status: {gif_status}", start_time)
    else:
        log("Skipping GIF assembly because --skip-gifs was set.", start_time)

    best_mem = max(eval_history, key=lambda row: row["f_mem"])
    best_quality = min(eval_history, key=lambda row: row["mmd_rbf"])
    summary = {
        "config": vars(args),
        "n_parameters": int(
            sum(parameter.numel() for parameter in flow_model.parameters())
        ),
        "final_loss": float(loss_history[-1]["loss"]) if loss_history else float("nan"),
        "best_memorization_step": int(best_mem["step"]),
        "best_memorization_f_mem": float(best_mem["f_mem"]),
        "best_quality_step": int(best_quality["step"]),
        "best_quality_mmd": float(best_quality["mmd_rbf"]),
        "final_mean_mem_ratio": float(eval_history[-1]["mean_mem_ratio"]),
        "final_mean_nn_ratio": float(eval_history[-1]["mean_nn_ratio"]),
        "final_f_mem": float(eval_history[-1]["f_mem"]),
        "final_mmd_rbf": float(eval_history[-1]["mmd_rbf"]),
        "final_chamfer": float(eval_history[-1]["chamfer"]),
        "baseline_mean_mem_ratio": float(eval_history[-1]["baseline_mean_mem_ratio"]),
        "baseline_mean_nn_ratio": float(eval_history[-1]["baseline_mean_ratio"]),
        "baseline_f_mem": float(eval_history[-1]["baseline_f_mem"]),
        "saved_snapshots": [int(row["step"]) for row in eval_history],
        "gif_status": gif_status,
    }
    save_metrics(summary, str(root / "summary.json"))
    log("Saved final summary.json.", start_time)

    print("=" * 80)
    print("Spiral memorization demo complete.")
    print(f"Output directory: {root}")
    print(
        f"Saved {len(eval_history)} evaluation snapshots and {len(loss_history)} training losses."
    )
    print(f"Final mean mem-ratio: {eval_history[-1]['mean_mem_ratio']:.4f}")
    print(f"Final f_mem:         {eval_history[-1]['f_mem']:.4f}")
    print(f"Final MMD:           {eval_history[-1]['mmd_rbf']:.4f}")
    print(f"Final Chamfer:       {eval_history[-1]['chamfer']:.4f}")
    print(f"Best quality step:   {best_quality['step']}")
    print(f"Best mem step:       {best_mem['step']}")
    if args.skip_gifs:
        print("GIF assembly was skipped; PNG frames are still saved.")
    else:
        print(f"GIF creation status: {gif_status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
