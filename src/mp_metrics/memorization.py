"""
Canonical memorization metrics for first-party studies.

This module owns two closely related interfaces:
- direct tensor-vs-tensor metrics for notebooks, sanity scripts, and loaded
  checkpoint outputs
- folder-vs-folder metrics for already-saved tensors or images on disk

The shared idea is always the same:
    1. compute first- and second-nearest-neighbor distances
    2. form the ratio `r = d1 / d2`
    3. declare a sample memorized when `r < threshold`
"""

from pathlib import Path

import numpy as np
import torch


# =============================================================================
# Direct tensor-level memorization metrics
# =============================================================================


def _flatten_batch(tensor):
    """
    Convert any batch-shaped tensor into `[B, D]` for distance computation.
    """
    if tensor.dim() == 2:
        return tensor
    return tensor.flatten(1)


def _resolve_ratio_threshold(ratio_threshold=None, config=None):
    """
    Resolve the active memorization threshold in one place.

    Priority:
        1. explicit `ratio_threshold=...`
        2. `config.evaluation.ratio_threshold`
        3. canonical default `0.5`

    This keeps the threshold explicit and controllable whether we are:
        - comparing tensors directly in a notebook
        - evaluating checkpoints from a run directory
        - comparing one folder of saved samples against another
    """
    if ratio_threshold is not None:
        return float(ratio_threshold)
    if config is not None and hasattr(config, "evaluation"):
        evaluation = config.evaluation
        if hasattr(evaluation, "ratio_threshold"):
            return float(evaluation.ratio_threshold)
    return 0.5


def compute_nn_stats(samples, reference, chunk_size=256):
    """
    Compute 1-NN, 2-NN, and the memorization ratio `r = d1 / d2`.

    Args:
        samples:   batch of generated or queried samples
        reference: batch of reference samples, usually the training set
        chunk_size: batch size for the distance computation

    Returns:
        `d1`, `d2`, `r`, `ratios`, `nn_idx`, and `indices`

    We return both the concise names (`r`, `nn_idx`) and the more explicit
    aliases (`ratios`, `indices`) so direct notebook use stays ergonomic while
    the checkpoint pipeline keeps its current field names.
    """
    samples = _flatten_batch(samples).float()
    reference = _flatten_batch(reference).float()

    if reference.shape[0] < 2:
        raise ValueError("Memorization metrics require at least two reference samples.")

    d1_list = []
    d2_list = []
    i1_list = []

    for start in range(0, samples.shape[0], chunk_size):
        chunk = samples[start : start + chunk_size]
        distances = torch.cdist(chunk, reference)
        topk = torch.topk(distances, k=2, largest=False, dim=1)
        d1_list.append(topk.values[:, 0])
        d2_list.append(topk.values[:, 1])
        i1_list.append(topk.indices[:, 0])

    d1 = torch.cat(d1_list)
    d2 = torch.cat(d2_list)
    nn_idx = torch.cat(i1_list)
    r = d1 / d2.clamp(min=1.0e-12)

    return {
        "d1": d1,
        "d2": d2,
        "r": r,
        "ratios": r,
        "nn_idx": nn_idx,
        "indices": nn_idx,
    }


def compute_f_mem(r, threshold=None, config=None):
    """
    Compute the memorization fraction from NN ratios.

    This is the scalar summary:
        f_mem = mean(r < threshold)

    The threshold is deliberately configurable everywhere so we can:
        - do sensitivity checks
        - align with a specific paper convention
        - keep sanity notebooks honest about what threshold they are using
    """
    threshold = _resolve_ratio_threshold(threshold, config=config)
    return float((r < threshold).float().mean().item())


def compute_memorization_metrics(
    samples,
    reference,
    ratio_threshold=None,
    chunk_size=256,
    config=None,
):
    """
    Canonical tensor-vs-tensor memorization interface.

    Use this when both datasets already live in memory as tensors, for example:
        - notebook cells
        - sanity scripts
        - checkpoint evaluation after loading saved tensors
    """
    threshold = _resolve_ratio_threshold(ratio_threshold, config=config)
    stats = compute_nn_stats(samples, reference, chunk_size=chunk_size)
    ratios = stats["r"]

    return {
        "ratio_threshold": threshold,
        "n_samples": int(samples.shape[0]),
        "n_reference": int(reference.shape[0]),
        "f_mem": compute_f_mem(ratios, threshold=threshold),
        "mean_nn_ratio": float(ratios.mean().item()),
        "std_nn_ratio": float(ratios.std(unbiased=False).item()),
        "mean_d1": float(stats["d1"].mean().item()),
        "mean_d2": float(stats["d2"].mean().item()),
    }


# =============================================================================
# Folder-level utilities
# =============================================================================


def _load_tensor_payload(path):
    """
    Load a tensor collection from a `.pt` or `.pth` payload.

    We accept either a raw tensor or a small dict containing a tensor under one
    of a few conventional keys so sanity scripts do not all have to serialize
    data in exactly the same way.
    """
    payload = torch.load(path, map_location="cpu")

    if torch.is_tensor(payload):
        return payload

    if isinstance(payload, dict):
        for key in ("samples", "tensor", "images", "data"):
            value = payload.get(key)
            if torch.is_tensor(value):
                return value

    raise ValueError(
        f"Could not find a tensor payload in {path}. "
        "Expected a tensor or a dict with one of: samples, tensor, images, data."
    )


def _load_image_payload(path):
    """
    Load one image file into a tensor without applying study-specific transforms.

    Folder-level memorization checks are meant to be lightweight and explicit,
    so this helper only performs the minimal conversion from image file to a
    channel-first tensor.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required for folder-based image memorization metrics."
        ) from exc

    image = Image.open(path)
    array = np.array(image, dtype=np.float32)

    if array.ndim == 2:
        tensor = torch.from_numpy(array).unsqueeze(0)
    elif array.ndim == 3:
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image rank in {path}: {array.ndim}")

    return tensor


def _normalize_loaded_tensor(tensor, source_path):
    """
    Ensure that loaded tensors carry an explicit batch dimension.

    A single image file naturally loads as `[C, H, W]`, while a saved tensor
    batch may already be `[B, ...]`. We normalize both cases so later code can
    concatenate collections without guessing which shape convention it got.
    """
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3 and source_path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }:
        return tensor.unsqueeze(0)
    return tensor


def _load_collection(path):
    """
    Load either one file or an entire directory tree into a batch tensor.

    Supported sources:
        - tensor files: `.pt`, `.pth`
        - image files: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            tensor = _load_tensor_payload(path)
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            tensor = _load_image_payload(path)
        else:
            raise ValueError(f"Unsupported file type for memorization metrics: {path}")
        return _normalize_loaded_tensor(tensor, path)

    tensors = []
    file_paths = sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower()
        in {".pt", ".pth", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    if not file_paths:
        raise FileNotFoundError(f"No supported tensor or image files found under {path}")

    for file_path in file_paths:
        if file_path.suffix.lower() in {".pt", ".pth"}:
            tensor = _load_tensor_payload(file_path)
        else:
            tensor = _load_image_payload(file_path)
        tensors.append(_normalize_loaded_tensor(tensor, file_path))

    return torch.cat(tensors, dim=0)


def compute_folder_metrics(
    samples_path,
    reference_path,
    ratio_threshold=None,
    chunk_size=256,
    config=None,
):
    """
    Canonical folder-vs-folder memorization interface.

    Supported contents:
        - tensor files: `.pt`, `.pth`
        - image files: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

    This is meant for cases where samples and references already exist on disk
    and you want to evaluate them without writing a one-off loader.
    """
    samples = _load_collection(samples_path)
    reference = _load_collection(reference_path)

    metrics = compute_memorization_metrics(
        samples,
        reference,
        ratio_threshold=ratio_threshold,
        chunk_size=chunk_size,
        config=config,
    )
    metrics["samples_path"] = str(Path(samples_path))
    metrics["reference_path"] = str(Path(reference_path))
    return metrics


if __name__ == "__main__":
    torch.manual_seed(0)

    print("mp_metrics.memorization example")

    # Treat `reference` as the training set and `samples` as nearby generations.
    reference = torch.randn(16, 8)
    samples = reference[:6] + 0.05 * torch.randn(6, 8)

    # Compute nearest-neighbor statistics first, then the thresholded summary.
    stats = compute_nn_stats(samples, reference)
    f_mem = compute_f_mem(stats["r"], threshold=0.5)
    metrics = compute_memorization_metrics(
        samples,
        reference,
        ratio_threshold=0.5,
    )

    # These prints are the basic sanity checks: ratio tensor shape, neighbor
    # indices, and the final memorization summary values.
    print(f"r shape              : {tuple(stats['r'].shape)}")
    print(f"nn indices shape     : {tuple(stats['indices'].shape)}")
    print(f"f_mem                : {f_mem:.6f}")
    print(f"metric keys          : {sorted(metrics.keys())}")
    print(f"mean_nn_ratio        : {metrics['mean_nn_ratio']:.6f}")
