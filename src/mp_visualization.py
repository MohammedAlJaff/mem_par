"""
Small image visualization helpers shared by training and Image32 evaluation.
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
    if figure_path.parent.name == "figures":
        return figure_path.parent.parent / "logs"
    return figure_path.parent / "logs"


def save_image_rows_figure(rows, row_labels, figure_path, *, title=None):
    """
    Save an image grid with one semantic row per tensor batch.
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


__all__ = [
    "is_image_batch",
    "maybe_display_figure",
    "model_context_title",
    "normalize_image_batch_for_display",
    "prepare_matplotlib_cache",
    "save_image_rows_figure",
    "save_samples_over_train_panel",
]
