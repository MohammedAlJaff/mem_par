"""
Notebook helpers for inspecting GDFM network setup.

This module is intentionally notebook-facing. It keeps the teaching notebook
small and readable while giving us a consistent way to inspect:
    - parameter counts
    - parameter grouping
    - layer-wise forward shapes
    - simple vector/image input-output visualizations

The goal is sanity checking and pedagogy, not benchmark-grade profiling.
"""

import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def format_params(count):
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def parameter_counts(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
    }


def parameter_overview(model):
    counts = parameter_counts(model)
    rows = []
    for key, value in counts.items():
        rows.append(
            {
                "metric": key,
                "count": value,
                "pretty": format_params(value),
            }
        )
    return pd.DataFrame(rows)


def _name_prefix(name, depth):
    if not name:
        return "<root>"
    parts = name.split(".")
    return ".".join(parts[: min(depth, len(parts))])


def parameter_groups(model, depth=2):
    grouped = {}
    for name, param in model.named_parameters():
        key = _name_prefix(name, depth)
        grouped[key] = grouped.get(key, 0) + param.numel()

    rows = []
    for name, count in grouped.items():
        rows.append(
            {
                "group": name,
                "params": count,
                "pretty": format_params(count),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("params", ascending=False).reset_index(drop=True)


def parameter_table(model):
    rows = []
    for name, param in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "params": param.numel(),
                "pretty": format_params(param.numel()),
                "trainable": bool(param.requires_grad),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("params", ascending=False).reset_index(drop=True)


def _shape_string(value):
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_shape_string(item) for item in value) + "]"
    if isinstance(value, dict):
        items = [f"{key}: {_shape_string(item)}" for key, item in value.items()]
        return "{" + ", ".join(items) + "}"
    return type(value).__name__


def run_with_trace(model, z, t, leaf_only=True, skip_parameterless=True):
    """
    Run a forward pass and capture a lightweight layer-by-layer shape trace.

    The trace records modules in call order and is best used as a sanity check.
    It is not a substitute for a profiler.
    """
    records = []
    hooks = []

    for name, module in model.named_modules():
        if name == "":
            continue
        if leaf_only and any(module.children()):
            continue

        direct_params = sum(
            param.numel() for param in module.parameters(recurse=False)
        )
        if skip_parameterless and direct_params == 0:
            continue

        def hook(current_module, inputs, outputs, current_name=name, current_params=direct_params):
            records.append(
                {
                    "module": current_name,
                    "class": current_module.__class__.__name__,
                    "input_shape": _shape_string(inputs),
                    "output_shape": _shape_string(outputs),
                    "direct_params": current_params,
                    "pretty": format_params(current_params),
                }
            )

        hooks.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(z, t)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    trace = pd.DataFrame(records)
    return output, trace


def tensor_stats(tensor, name="tensor"):
    return {
        "name": name,
        "shape": tuple(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "l2_norm": float(tensor.norm().item()),
    }


def stats_table(pairs):
    return pd.DataFrame([tensor_stats(tensor, name) for name, tensor in pairs])


def make_vector_demo(data_dim, batch_size=1):
    axis = torch.linspace(-1.0, 1.0, data_dim)
    wave = 0.35 * torch.sin(torch.linspace(0.0, 4.0 * math.pi, data_dim))
    bump = torch.exp(-((axis - 0.2) ** 2) / 0.04)
    vector = axis + wave + 0.45 * (bump - bump.mean())
    return vector.unsqueeze(0).repeat(batch_size, 1)


def make_grayscale_demo(image_size, batch_size=1):
    grid = torch.linspace(-1.0, 1.0, image_size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    radius = torch.sqrt(xx**2 + yy**2)
    ring = torch.exp(-((radius - 0.55) ** 2) / 0.015)
    stripe = 0.30 * torch.sin(3.0 * math.pi * xx) * torch.cos(2.0 * math.pi * yy)
    square = (((xx.abs() < 0.35) & (yy.abs() < 0.22)).float() * 0.50)
    image = (0.55 * ring + stripe + square).clamp(-1.0, 1.0)
    return image.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)


def make_rgb_demo(image_size, batch_size=1):
    grid = torch.linspace(-1.0, 1.0, image_size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    red = xx
    green = torch.exp(-((xx + 0.25) ** 2 + (yy - 0.10) ** 2) / 0.18)
    blue = 0.60 * torch.sin(math.pi * xx) * torch.cos(math.pi * yy)
    image = torch.stack([red, green, blue], dim=0).clamp(-1.0, 1.0)
    return image.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def make_time_batch(batch_size, value=0.35):
    return torch.full((batch_size,), float(value))


def _prepare_gray(image):
    return image.detach().cpu().squeeze().clamp(-1.0, 1.0)


def _prepare_rgb(image):
    image = image.detach().cpu().permute(1, 2, 0)
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)


def plot_vector_io(z, y, example_idx=0):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
    axis = torch.arange(z.shape[1]).cpu()
    input_vec = z[example_idx].detach().cpu()
    output_vec = y[example_idx].detach().cpu()

    axes[0].plot(axis, input_vec, linewidth=2.0)
    axes[0].set_title("Input Vector")
    axes[0].set_xlabel("feature index")
    axes[0].set_ylabel("value")
    axes[0].grid(alpha=0.3)

    axes[1].plot(axis, output_vec, linewidth=2.0, color="tab:orange")
    axes[1].set_title("Output Vector")
    axes[1].set_xlabel("feature index")
    axes[1].set_ylabel("value")
    axes[1].grid(alpha=0.3)

    return fig


def plot_image_io(z, y, example_idx=0):
    channels = z.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    if channels == 1:
        input_image = _prepare_gray(z[example_idx])
        output_image = _prepare_gray(y[example_idx])
        diff_image = (output_image - input_image).clamp(-1.0, 1.0)

        axes[0].imshow(input_image, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[1].imshow(output_image, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[2].imshow(diff_image, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    elif channels == 3:
        input_image = _prepare_rgb(z[example_idx])
        output_image = _prepare_rgb(y[example_idx])
        diff_image = (y[example_idx] - z[example_idx]).detach().cpu().pow(2).sum(0).sqrt()

        axes[0].imshow(input_image)
        axes[1].imshow(output_image)
        axes[2].imshow(diff_image, cmap="magma")
    else:
        raise ValueError(f"plot_image_io expects 1 or 3 channels, got {channels}")

    axes[0].set_title("Input")
    axes[1].set_title("Output")
    axes[2].set_title("Change Magnitude")

    for axis in axes:
        axis.axis("off")

    return fig


def backbone_row(name, model, z, y):
    counts = parameter_counts(model)
    return {
        "backbone": name,
        "class": model.__class__.__name__,
        "inner_impl": getattr(getattr(model, "model", None), "__class__", type(None)).__name__,
        "input_shape": tuple(z.shape),
        "output_shape": tuple(y.shape),
        "params": counts["total_params"],
        "pretty": format_params(counts["total_params"]),
    }


__all__ = [
    "backbone_row",
    "format_params",
    "make_grayscale_demo",
    "make_rgb_demo",
    "make_time_batch",
    "make_vector_demo",
    "parameter_counts",
    "parameter_groups",
    "parameter_overview",
    "parameter_table",
    "plot_image_io",
    "plot_vector_io",
    "run_with_trace",
    "set_seed",
    "stats_table",
    "tensor_stats",
]
