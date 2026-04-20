"""
Checkpoint sample preview helpers shared by evaluation paths.
"""

from pathlib import Path

import torch

from mp_visualization import (
    is_image_batch,
    model_context_title,
    save_samples_over_train_panel,
)


DEFAULT_CHECKPOINT_PREVIEW_COUNT = 8


def checkpoint_preview_figure_name(step):
    """Return the stable figure key for one checkpoint preview panel."""
    return f"checkpoint_samples_over_train_step_{int(step):06d}"


def _visual_sanity_config(config):
    visual = getattr(config, "visual_sanity", None) if config is not None else None
    if visual is None or not bool(getattr(visual, "enabled", False)):
        return None
    return visual


def _preview_count(config, samples, train_tensor=None, count=None):
    if count is None:
        visual = _visual_sanity_config(config)
        if visual is not None:
            count = int(getattr(visual, "num_images", DEFAULT_CHECKPOINT_PREVIEW_COUNT))
        else:
            evaluation = getattr(config, "evaluation", None) if config is not None else None
            count = int(
                getattr(
                    evaluation,
                    "sample_panel_count",
                    DEFAULT_CHECKPOINT_PREVIEW_COUNT,
                )
            )

    limit = int(samples.shape[0])
    if train_tensor is not None:
        limit = min(limit, int(train_tensor.shape[0]))
    return max(0, min(int(count), limit))


def collect_checkpoint_sample_preview_record(
    *,
    step,
    samples,
    train_tensor,
    run_dir,
    config=None,
    count=None,
    figure_enabled=None,
):
    """
    Save a compact sample preview tensor and, for image data, an optional panel.
    """
    step = int(step)
    run_dir = Path(run_dir)
    count = _preview_count(config, samples, train_tensor=train_tensor, count=count)
    samples_dir = run_dir / "metrics" / "checkpoint_sample_previews"
    samples_path = samples_dir / f"step_{step:06d}.pt"
    samples_dir.mkdir(parents=True, exist_ok=True)

    preview_samples = samples[:count].detach().cpu()
    torch.save(preview_samples, samples_path)

    figure_path = None
    should_render = (
        bool(figure_enabled)
        if figure_enabled is not None
        else _visual_sanity_config(config) is not None
    )
    if (
        should_render
        and count > 0
        and train_tensor is not None
        and is_image_batch(preview_samples)
        and is_image_batch(train_tensor)
    ):
        figure_name = checkpoint_preview_figure_name(step)
        saved_path = save_samples_over_train_panel(
            preview_samples,
            train_tensor,
            run_dir / "figures" / f"{figure_name}.png",
            count=count,
            title=(
                model_context_title(
                    config,
                    "Checkpoint samples over train images",
                    checkpoint_step=step,
                )
                if config is not None
                else f"Checkpoint samples over train images\ncheckpoint_step={step}"
            ),
        )
        if saved_path is not None:
            figure_path = str(Path(saved_path).resolve())

    return {
        "step": step,
        "count": int(count),
        "samples_path": str(samples_path.resolve()),
        "figure_path": figure_path,
    }


__all__ = [
    "DEFAULT_CHECKPOINT_PREVIEW_COUNT",
    "checkpoint_preview_figure_name",
    "collect_checkpoint_sample_preview_record",
]
