"""
File and folder helpers for ad hoc metric evaluation.

Metric modules accept tensors. These helpers load tensors or image files, then
call the tensor-based metric functions.
"""

from pathlib import Path

import numpy as np
import torch

from mp_metrics.images import (
    DEFAULT_FID_FEATURE_EXTRACTOR,
    compute_image_fid,
    compute_image_memorization_metrics,
)
from mp_metrics.synthetic import compute_memorization_metrics


def _load_tensor_payload(path):
    """Load a tensor payload from a `.pt` or `.pth` file."""
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, dict):
        for key in ("samples", "tensor", "images", "data"):
            value = payload.get(key)
            if torch.is_tensor(value):
                return value
    raise ValueError(f"Could not find a tensor payload in {path}.")


def _load_image_payload(path):
    """Load one image file into a channel-first tensor."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required for image files.") from exc

    image = Image.open(path)
    array = np.array(image, dtype=np.float32)
    if array.ndim == 2:
        return torch.from_numpy(array).unsqueeze(0)
    if array.ndim == 3:
        return torch.from_numpy(array).permute(2, 0, 1)
    raise ValueError(f"Unsupported image rank in {path}: {array.ndim}")


def _normalize_loaded_tensor(tensor, source_path):
    """Add a batch dimension when a loaded file contains one sample."""
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


def load_metric_collection(path):
    """Load tensor or image files from one path into a batch tensor."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    tensor_suffixes = {".pt", ".pth"}
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in tensor_suffixes:
            tensor = _load_tensor_payload(path)
        elif suffix in image_suffixes:
            tensor = _load_image_payload(path)
        else:
            raise ValueError(f"Unsupported file type for metrics: {path}")
        return _normalize_loaded_tensor(tensor, path)

    file_paths = sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in tensor_suffixes.union(image_suffixes)
    )
    if not file_paths:
        raise FileNotFoundError(f"No supported tensor or image files found under {path}")

    tensors = []
    for file_path in file_paths:
        if file_path.suffix.lower() in tensor_suffixes:
            tensor = _load_tensor_payload(file_path)
        else:
            tensor = _load_image_payload(file_path)
        tensors.append(_normalize_loaded_tensor(tensor, file_path))
    return torch.cat(tensors, dim=0)


def compute_synthetic_folder_metrics(
    samples_path,
    reference_path,
    ratio_threshold=None,
    chunk_size=256,
    config=None,
):
    """Load two collections and compute synthetic nearest-neighbor metrics."""
    samples = load_metric_collection(samples_path)
    reference = load_metric_collection(reference_path)
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


def compute_image_folder_memorization_metrics(
    samples_path,
    reference_path,
    *,
    encoder=None,
    rho=None,
    query_batch_size=256,
    reference_batch_size=None,
    image_range="auto",
    device="cpu",
):
    """Load two image/tensor folders and compute image memorization metrics."""
    samples = load_metric_collection(samples_path)
    reference = load_metric_collection(reference_path)
    metrics = compute_image_memorization_metrics(
        samples,
        reference,
        encoder=encoder,
        rho=rho,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
        image_range=image_range,
        device=device,
    )
    metrics["samples_path"] = str(Path(samples_path))
    metrics["reference_path"] = str(Path(reference_path))
    return metrics


def compute_image_folder_fid(
    samples_path,
    reference_path=None,
    *,
    reference_stats_path=None,
    batch_size=64,
    device="cuda",
    image_range="auto",
    feature_extractor=DEFAULT_FID_FEATURE_EXTRACTOR,
    cache=False,
    verbose=False,
):
    """Load image/tensor folders and compute FID."""
    samples = load_metric_collection(samples_path)
    reference = load_metric_collection(reference_path) if reference_path is not None else None
    metrics = compute_image_fid(
        samples,
        reference,
        reference_stats_path=reference_stats_path,
        batch_size=batch_size,
        device=device,
        image_range=image_range,
        feature_extractor=feature_extractor,
        cache=cache,
        verbose=verbose,
    )
    metrics["samples_path"] = str(Path(samples_path))
    metrics["reference_path"] = None if reference_path is None else str(Path(reference_path))
    return metrics


__all__ = [
    "compute_image_folder_fid",
    "compute_image_folder_memorization_metrics",
    "compute_synthetic_folder_metrics",
    "load_metric_collection",
]


if __name__ == "__main__":
    print("mp_runners.metric_io example")
    print("Use load_metric_collection(path) before tensor-based metric calls.")
