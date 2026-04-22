"""
Visualization helpers shared by training and evaluation.

These helpers turn saved tensors and metric records into PNG figures. They do
not compute metric values; they make saved evidence easier to inspect by eye.
"""

import os
from pathlib import Path

import torch


def prepare_matplotlib_cache(logs_dir):
    """
    Keep matplotlib from trying to write cache files under an unwritable home.
    """
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = Path(logs_dir) / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def is_image_batch(tensor):
    """
    Check whether a tensor can be rendered as an image batch.
    """
    return (
        torch.is_tensor(tensor)
        and tensor.dim() == 4
        and int(tensor.shape[1]) in (1, 3, 4)
    )


def normalize_image_batch_for_display(tensor):
    """
    Convert common image tensor ranges to `[0, 1]` for matplotlib display.

    Model tensors often use `[-1, 1]`; matplotlib expects display pixels.
    """
    images = tensor.detach().float().cpu()
    if images.numel() == 0:
        raise ValueError("Image grids require at least one image.")

    min_value = float(images.min().item())
    max_value = float(images.max().item())
    if min_value >= 0.0 and max_value <= 1.0:
        return images.clamp(0.0, 1.0)
    if min_value >= 0.0 and max_value <= 255.0:
        return (images / 255.0).clamp(0.0, 1.0)
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def _config_value(config, section_name, field_name, fallback="?"):
    """
    Read a nested config value for figure titles.
    """
    section = getattr(config, section_name, None)
    if section is None:
        return fallback
    return getattr(section, field_name, fallback)


def model_context_title(config, heading, *, checkpoint_step=None, extra=None):
    """
    Build an informative title for saved image panels.
    """
    run_parts = []
    if checkpoint_step is not None:
        run_parts.append(f"checkpoint_step={int(checkpoint_step)}")
    if extra:
        run_parts.append(str(extra))
    run_parts.extend(
        [
            f"train_steps={_config_value(config, 'training', 'n_steps')}",
            f"sample_steps={_config_value(config, 'sampling', 'n_steps')}",
            (
                f"data={_config_value(config, 'data', 'dataset')}"
                f"@{_config_value(config, 'data', 'image_size')}"
            ),
        ]
    )
    model_parts = [
        f"framework={_config_value(config, 'framework', 'family')}",
        f"prediction_space={_config_value(config, 'framework', 'prediction_space')}",
        f"loss_space={_config_value(config, 'framework', 'loss_space')}",
        f"model={_config_value(config, 'network', 'arch')}",
    ]
    return "{}\n{}\n{}".format(
        str(heading),
        " | ".join(run_parts),
        " | ".join(model_parts),
    )


def _draw_image_on_axis(axis, image):
    """
    Draw one `[C, H, W]` image tensor onto a matplotlib axis.
    """
    if int(image.shape[0]) == 1:
        axis.imshow(image[0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    else:
        axis.imshow(image[:3].permute(1, 2, 0).numpy(), vmin=0.0, vmax=1.0)
    axis.set_xticks([])
    axis.set_yticks([])


def _logs_dir_for_figure(figure_path):
    """
    Return the logs directory associated with a saved figure path.
    """
    if figure_path.parent.name == "figures":
        return figure_path.parent.parent / "logs"
    return figure_path.parent / "logs"


def save_image_rows_figure(rows, row_labels, figure_path, *, title=None):
    """
    Save an image grid with one semantic row per tensor batch.

    Typical rows are generated samples and the corresponding training images.
    Keeping rows explicit makes figure meaning clear in run folders and W&B.
    """
    figure_path = Path(figure_path)
    prepare_matplotlib_cache(_logs_dir_for_figure(figure_path))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_rows = [normalize_image_batch_for_display(row) for row in rows]
    n_rows = len(normalized_rows)
    n_cols = max(int(row.shape[0]) for row in normalized_rows)
    title_lines = str(title).count("\n") + 1 if title else 0
    title_height = 0.32 * title_lines if title else 0.0
    fig_width = max(4.0, 1.25 * n_cols)
    fig_height = max(2.0, 1.45 * n_rows + title_height)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for row_index, images in enumerate(normalized_rows):
        for col_index in range(n_cols):
            axis = axes[row_index][col_index]
            axis.axis("off")
            if col_index < int(images.shape[0]):
                _draw_image_on_axis(axis, images[col_index])
                if col_index == 0:
                    axis.set_ylabel(
                        str(row_labels[row_index]),
                        rotation=0,
                        labelpad=38,
                        va="center",
                        ha="right",
                        fontsize=9,
                    )

    if title:
        fig.suptitle(str(title), fontsize=10.5)
        top = 0.86 if title_lines > 1 else 0.92
        fig.tight_layout(pad=0.45, rect=(0.0, 0.0, 1.0, top))
    else:
        fig.tight_layout(pad=0.45)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def save_samples_over_train_panel(
    samples,
    train_tensor,
    figure_path,
    *,
    count,
    title,
):
    """
    Save generated samples above train examples using the shared image style.
    """
    if not is_image_batch(samples) or not is_image_batch(train_tensor):
        return None
    count = min(int(count), int(samples.shape[0]), int(train_tensor.shape[0]))
    if count < 1:
        return None
    return save_image_rows_figure(
        [samples[:count], train_tensor[:count]],
        ["samples", "train"],
        figure_path,
        title=title,
    )


def maybe_display_figure(figure_path, enabled):
    """
    Display a figure inline when running in a notebook; CLI runs keep the path.
    """
    if not bool(enabled):
        return False
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ModuleNotFoundError:
        return False

    shell = get_ipython()
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return False
    display(Image(filename=str(figure_path)))
    return True


def make_synthetic_figures(
    *,
    figures_dir,
    train_history,
    metric_records,
    train_tensor,
    reference_tensor,
    final_samples,
    train_intrinsic,
    reference_intrinsic,
    final_samples_intrinsic,
    geometry=None,
    intrinsic_centers=None,
    ambient_centers=None,
    context="",
):
    """
    Create synthetic evaluation figures for one completed evaluation pass.

    Figures are derived from saved train/reference tensors, final generated
    samples, and metric records. They do not change metric values.
    """
    figure_paths = {}
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warning_path = Path(figures_dir) / "figure_generation_error.txt"
        warning_path.write_text("{}: {}".format(type(exc).__name__, exc), encoding="utf-8")
        return {"figure_generation_error": str(warning_path)}

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if train_history:
        figure_name = "training_loss_curve_by_optimization_step"
        path = figures_dir / f"{figure_name}.png"
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(
            [row["step"] for row in train_history],
            [row["loss"] for row in train_history],
            linewidth=1.8,
        )
        ax.set_title("Training loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths[figure_name] = str(path)

    if metric_records:
        _synthetic_metric_curves(plt, figures_dir, figure_paths, metric_records, context)

    geometry_kind = ""
    if isinstance(geometry, dict):
        geometry_kind = str(geometry.get("kind", ""))

    if train_intrinsic.shape[1] >= 2:
        if geometry_kind == "torus3d":
            scatter_title = "Generated samples in torus periodic angle features"
            x_label = "cos(theta)"
            y_label = "sin(theta)"
        else:
            scatter_title = "Generated samples in intrinsic coordinates"
            x_label = "intrinsic[0]"
            y_label = "intrinsic[1]"
        _synthetic_intrinsic_scatter(
            plt,
            figures_dir,
            figure_paths,
            train_intrinsic,
            reference_intrinsic,
            final_samples_intrinsic,
            intrinsic_centers,
            context,
            name="intrinsic_space_scatter_train_reference_generated_2d",
            title=scatter_title,
            x_label=x_label,
            y_label=y_label,
        )

    if geometry_kind == "torus3d":
        _synthetic_torus3d_scatters(
            plt,
            figures_dir,
            figure_paths,
            train_tensor,
            reference_tensor,
            final_samples,
            geometry,
            context,
        )

    if train_tensor.shape[1] == 2:
        _synthetic_ambient_scatter(
            plt,
            figures_dir,
            figure_paths,
            train_tensor,
            reference_tensor,
            final_samples,
            ambient_centers,
            context,
        )

    return figure_paths


def _synthetic_metric_curves(plt, figures_dir, figure_paths, metric_records, context):
    """
    Save scalar synthetic metric curves over checkpoint step.

    Each curve compares generated samples with a held-out reference baseline.
    """
    steps = [row["step"] for row in metric_records]
    plot_specs = [
        (
            "nearest_neighbor_ratio_d1_over_d2_by_checkpoint_step",
            "Nearest-neighbor ratio d1/d2",
            "d1 / d2",
            [
                (
                    "generated ambient",
                    [row["ambient_nearest_neighbor_ratio_d1_over_d2"] for row in metric_records],
                    "-",
                ),
                (
                    "generated intrinsic",
                    [row["intrinsic_nearest_neighbor_ratio_d1_over_d2"] for row in metric_records],
                    "-",
                ),
                (
                    "holdout ambient",
                    [row["holdout_ambient_nearest_neighbor_ratio_d1_over_d2"] for row in metric_records],
                    "--",
                ),
                (
                    "holdout intrinsic",
                    [row["holdout_intrinsic_nearest_neighbor_ratio_d1_over_d2"] for row in metric_records],
                    "--",
                ),
            ],
        ),
        (
            "intrinsic_distance_vs_step_checkpoints",
            "Intrinsic distribution distances",
            "distance",
            [
                ("intrinsic sliced-W2", [row["intrinsic_sliced_w2"] for row in metric_records], "-"),
                (
                    "holdout intrinsic sliced-W2",
                    [row["holdout_intrinsic_sliced_w2"] for row in metric_records],
                    "--",
                ),
            ],
        ),
    ]
    if "intrinsic_gmm_nll_mean" in metric_records[-1]:
        plot_specs.append(
            (
                "intrinsic_gmm_negative_log_likelihood_vs_checkpoint_step",
                "Intrinsic GMM negative log-likelihood",
                "NLL",
                [
                    (
                        "generated intrinsic NLL",
                        [row["intrinsic_gmm_nll_mean"] for row in metric_records],
                        "-",
                    ),
                    (
                        "holdout intrinsic NLL",
                        [row["holdout_intrinsic_gmm_nll_mean"] for row in metric_records],
                        "--",
                    ),
                ],
            )
        )

    for name, title, ylabel, series in plot_specs:
        path = figures_dir / "{}.png".format(name)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for label, values, linestyle in series:
            if any(value is not None for value in values):
                ax.plot(steps, values, marker="o", linestyle=linestyle, label=label)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths[name] = str(path)


def _synthetic_intrinsic_scatter(
    plt,
    figures_dir,
    figure_paths,
    train,
    reference,
    samples,
    centers,
    context,
    *,
    name,
    title,
    x_label,
    y_label,
):
    """Save a 2D intrinsic-coordinate scatter plot."""
    pieces = [train[:, :2], reference[:, :2], samples[:, :2]]
    if centers is not None and centers.shape[1] >= 2:
        pieces.append(centers[:, :2])
    limits = torch.cat(pieces, dim=0)
    x_min = float(limits[:, 0].min().item()) - 0.5
    x_max = float(limits[:, 0].max().item()) + 0.5
    y_min = float(limits[:, 1].min().item()) - 0.5
    y_max = float(limits[:, 1].max().item()) + 0.5

    path = figures_dir / "{}.png".format(name)
    fig, ax = plt.subplots(figsize=(5.8, 6.1))
    ax.scatter(train[:, 0].numpy(), train[:, 1].numpy(), s=24, alpha=0.25, label="train")
    ax.scatter(reference[:, 0].numpy(), reference[:, 1].numpy(), s=8, alpha=0.20, label="reference")
    ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=14, alpha=0.75, label="generated")
    if centers is not None and centers.shape[1] >= 2:
        ax.scatter(centers[:, 0].numpy(), centers[:, 1].numpy(), s=90, marker="x", linewidths=2.0, label="centers")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths[name] = str(path)


def _synthetic_torus3d_canonical_parts(train_tensor, reference_tensor, samples, geometry):
    """Return canonical 3D torus coordinates for train, reference, and samples."""
    from mp_metrics.synthetic import intrinsic_geometry_parts

    train_xyz = geometry.get("train_torus_xyz")
    if train_xyz is None:
        train_xyz = intrinsic_geometry_parts(train_tensor, geometry)["torus_xyz"]
    reference_xyz = geometry.get("reference_torus_xyz")
    if reference_xyz is None:
        reference_xyz = intrinsic_geometry_parts(reference_tensor, geometry)["torus_xyz"]
    sample_xyz = intrinsic_geometry_parts(samples, geometry)["torus_xyz"]
    return train_xyz.cpu(), reference_xyz.cpu(), sample_xyz.cpu()


def _set_equal_3d_limits(ax, pieces):
    """Set equal 3D axes for a list of point tensors."""
    points = torch.cat(pieces, dim=0)
    mins = points.min(dim=0).values
    maxs = points.max(dim=0).values
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float((maxs - mins).max().item())
    radius = max(radius, 1.0e-6)
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))


def _synthetic_torus3d_scatters(
    plt,
    figures_dir,
    figure_paths,
    train_tensor,
    reference_tensor,
    samples,
    geometry,
    context,
):
    """Save canonical 3D torus scatter plots."""
    train_xyz, reference_xyz, sample_xyz = _synthetic_torus3d_canonical_parts(
        train_tensor,
        reference_tensor,
        samples,
        geometry,
    )
    pieces = [train_xyz, reference_xyz, sample_xyz]

    path_3d_name = "torus3d_canonical_scatter_train_reference_generated_3d"
    path_3d = figures_dir / f"{path_3d_name}.png"
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(train_xyz[:, 0].numpy(), train_xyz[:, 1].numpy(), train_xyz[:, 2].numpy(), s=12, alpha=0.18, label="train")
    ax.scatter(reference_xyz[:, 0].numpy(), reference_xyz[:, 1].numpy(), reference_xyz[:, 2].numpy(), s=8, alpha=0.12, label="holdout")
    ax.scatter(sample_xyz[:, 0].numpy(), sample_xyz[:, 1].numpy(), sample_xyz[:, 2].numpy(), s=14, alpha=0.72, label="generated")
    ax.set_title("Generated samples on canonical 3D torus")
    ax.set_xlabel("torus x")
    ax.set_ylabel("torus y")
    ax.set_zlabel("torus z")
    _set_equal_3d_limits(ax, pieces)
    ax.legend()
    fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path_3d, dpi=160)
    plt.close(fig)
    figure_paths[path_3d_name] = str(path_3d)

    path_xy_name = "torus3d_xy_projection_scatter_train_reference_generated"
    path_xy = figures_dir / f"{path_xy_name}.png"
    xy_points = torch.cat([piece[:, :2] for piece in pieces], dim=0)
    x_min = float(xy_points[:, 0].min().item()) - 0.25
    x_max = float(xy_points[:, 0].max().item()) + 0.25
    y_min = float(xy_points[:, 1].min().item()) - 0.25
    y_max = float(xy_points[:, 1].max().item()) + 0.25
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(train_xyz[:, 0].numpy(), train_xyz[:, 1].numpy(), s=14, alpha=0.18, label="train")
    ax.scatter(reference_xyz[:, 0].numpy(), reference_xyz[:, 1].numpy(), s=8, alpha=0.12, label="holdout")
    ax.scatter(sample_xyz[:, 0].numpy(), sample_xyz[:, 1].numpy(), s=14, alpha=0.72, label="generated")
    ax.set_title("Canonical torus XY projection")
    ax.set_xlabel("torus x")
    ax.set_ylabel("torus y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path_xy, dpi=160)
    plt.close(fig)
    figure_paths[path_xy_name] = str(path_xy)


def _synthetic_ambient_scatter(plt, figures_dir, figure_paths, train, reference, samples, centers, context):
    """Save a 2D ambient-space scatter plot."""
    pieces = [train[:, :2], reference[:, :2], samples[:, :2]]
    if centers is not None and centers.shape[1] >= 2:
        pieces.append(centers[:, :2])
    points = torch.cat(pieces, dim=0)
    x_min = float(points[:, 0].min().item()) - 0.5
    x_max = float(points[:, 0].max().item()) + 0.5
    y_min = float(points[:, 1].min().item()) - 0.5
    y_max = float(points[:, 1].max().item()) + 0.5

    path_name = "ambient_space_scatter_train_generated_2d"
    path = figures_dir / f"{path_name}.png"
    fig, ax = plt.subplots(figsize=(5.8, 6.1))
    ax.scatter(train[:, 0].numpy(), train[:, 1].numpy(), s=24, alpha=0.25, label="train")
    ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=14, alpha=0.75, label="generated")
    if centers is not None and centers.shape[1] >= 2:
        ax.scatter(centers[:, 0].numpy(), centers[:, 1].numpy(), s=90, marker="x", linewidths=2.0, label="ambient centers")
    ax.set_title("Generated samples in ambient coordinates")
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths[path_name] = str(path)


__all__ = [
    "is_image_batch",
    "make_synthetic_figures",
    "maybe_display_figure",
    "model_context_title",
    "normalize_image_batch_for_display",
    "prepare_matplotlib_cache",
    "save_image_rows_figure",
    "save_samples_over_train_panel",
]


if __name__ == "__main__":
    print("mp_visualization example")
    print("is_image_batch:", is_image_batch(torch.zeros(1, 1, 4, 4)))
