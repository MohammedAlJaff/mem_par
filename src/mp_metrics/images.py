"""
Image-facing memorization and FID metrics.

This module keeps real-image evaluation explicit:
- image tensors are normalized to `[0, 1]` here, not in the generic tensor
  memorization helpers
- feature encoders run in batches and return CPU feature tensors
- FID goes through `torch_fidelity.calculate_metrics` directly
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from mp_metrics.memorization import (
    _load_collection,
    compute_f_mem,
    compute_memorization_metrics,
)


DEFAULT_FEATURE_ENCODER = "torch_fidelity:inception-v3-compat:2048"
DEFAULT_FID_FEATURE_EXTRACTOR = "inception-v3-compat"


class _TensorImageDataset(Dataset):
    """Small torch-fidelity adapter for an in-memory uint8 RGB image batch."""

    def __init__(self, images):
        if not torch.is_tensor(images):
            raise TypeError("images must be a torch.Tensor")
        if images.dtype != torch.uint8:
            raise TypeError(f"FID images must be uint8, got {images.dtype}")
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"FID images must have shape [B, 3, H, W], got {tuple(images.shape)}")
        self.images = images.cpu().contiguous()

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, index):
        return self.images[index]


def _as_image_batch(images):
    """Normalize common image tensor layouts to `[B, C, H, W]`."""
    if not torch.is_tensor(images):
        images = torch.as_tensor(images)
    images = images.detach()

    if images.dim() == 2:
        return images.unsqueeze(0).unsqueeze(0)
    if images.dim() == 3:
        if images.shape[0] in (1, 3, 4):
            return images.unsqueeze(0)
        if images.shape[-1] in (1, 3, 4):
            return images.permute(2, 0, 1).unsqueeze(0)
        return images.unsqueeze(1)
    if images.dim() == 4:
        if images.shape[1] in (1, 3, 4):
            return images
        if images.shape[-1] in (1, 3, 4):
            return images.permute(0, 3, 1, 2)

    raise ValueError(
        "Expected image tensor shaped [B, C, H, W], [B, H, W], [C, H, W], "
        f"or [H, W], got {tuple(images.shape)}"
    )


def _normalize_images(images, image_range="auto"):
    """
    Convert image tensors to float `[0, 1]` with an explicit batch dimension.
    """
    images = _as_image_batch(images).float()
    mode = str(image_range).lower().replace("-", "_")

    if mode in {"0_1", "zero_one", "unit"}:
        return images.clamp(0.0, 1.0)
    if mode in {"minus_one_one", "neg_one_one", "-1_1", "_1_1"}:
        return ((images + 1.0) * 0.5).clamp(0.0, 1.0)
    if mode in {"0_255", "uint8", "byte"}:
        return (images / 255.0).clamp(0.0, 1.0)
    if mode != "auto":
        raise ValueError(
            "image_range must be one of auto, 0_1, -1_1, or 0_255; "
            f"got {image_range!r}"
        )

    if images.numel() == 0:
        raise ValueError("Image metrics require at least one image.")

    min_value = float(images.min().item())
    max_value = float(images.max().item())
    if min_value >= 0.0 and max_value <= 1.0:
        return images.clamp(0.0, 1.0)
    if min_value >= -1.0 and max_value <= 1.0:
        return ((images + 1.0) * 0.5).clamp(0.0, 1.0)
    if min_value >= 0.0 and max_value <= 255.0:
        return (images / 255.0).clamp(0.0, 1.0)

    raise ValueError(
        "Could not infer image range. Pass image_range='0_1', '-1_1', "
        f"or '0_255'. Observed min={min_value:.4g}, max={max_value:.4g}."
    )


def _to_uint8_rgb(images, image_range="auto"):
    """
    Convert image tensors to torch-fidelity's expected uint8 RGB `[B, 3, H, W]`.
    """
    images = _normalize_images(images, image_range=image_range)
    channels = int(images.shape[1])
    if channels == 1:
        images = images.repeat(1, 3, 1, 1)
    elif channels == 4:
        images = images[:, :3]
    elif channels != 3:
        raise ValueError(f"Expected 1, 3, or 4 image channels, got {channels}")

    return torch.round(images * 255.0).clamp(0, 255).to(torch.uint8).cpu()


def _device_from_arg(device):
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _uses_cuda(device):
    return _device_from_arg(device).type == "cuda"


def _select_cuda_device(device):
    torch_device = _device_from_arg(device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device.index)
    return torch_device


def _coerce_feature_output(output):
    if isinstance(output, dict):
        if "features" in output:
            output = output["features"]
        elif len(output) == 1:
            output = next(iter(output.values()))
        else:
            raise ValueError("Callable encoder returned a dict with multiple feature tensors.")
    elif isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise ValueError("Callable encoder must return one feature tensor.")
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError("Encoder output must be a torch.Tensor.")
    return output.flatten(1).float()


def _encode_with_callable(images, encoder, *, batch_size, image_range, device):
    images = _normalize_images(images, image_range=image_range)
    torch_device = _device_from_arg(device)

    was_training = getattr(encoder, "training", None)
    if hasattr(encoder, "eval"):
        encoder.eval()
    if hasattr(encoder, "to"):
        encoder.to(torch_device)

    features = []
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], batch_size):
                batch = images[start : start + batch_size].to(torch_device)
                encoded = _coerce_feature_output(encoder(batch))
                features.append(encoded.detach().cpu())
    finally:
        if was_training is not None and hasattr(encoder, "train"):
            encoder.train(was_training)

    return torch.cat(features, dim=0)


class _TorchFidelityFeatureEncoder:
    """Batched feature extractor wrapper for torch-fidelity encoders."""

    def __init__(self, spec, device):
        self.spec = spec
        self.device = _select_cuda_device(device)
        self.extractor_name, self.feature_layer = _parse_torch_fidelity_spec(spec)

        try:
            from torch_fidelity.utils import create_feature_extractor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch-fidelity is required for torch_fidelity:* image encoders."
            ) from exc

        self.extractor = create_feature_extractor(
            self.extractor_name,
            [self.feature_layer],
            cuda=self.device.type == "cuda",
            verbose=False,
        )
        self.extractor.eval()

    def encode(self, batch):
        batch = _to_uint8_rgb(batch, image_range="0_1").to(self.device)
        with torch.no_grad():
            output = self.extractor(batch)
        features = self.extractor.convert_features_tuple_to_dict(output)
        return features[self.feature_layer].flatten(1).float().cpu()


def _parse_torch_fidelity_spec(spec):
    parts = str(spec).split(":")
    if len(parts) == 2 and parts[0] == "torch_fidelity":
        extractor_name = parts[1]
        try:
            from torch_fidelity.utils import resolve_feature_layer_for_metric
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch-fidelity is required for torch_fidelity:* image encoders."
            ) from exc
        feature_layer = resolve_feature_layer_for_metric(
            "fid",
            feature_extractor=extractor_name,
        )
        return extractor_name, feature_layer
    if len(parts) == 3 and parts[0] == "torch_fidelity":
        return parts[1], parts[2]
    raise ValueError(
        "Encoder spec must look like "
        "'torch_fidelity:inception-v3-compat:2048'."
    )


def _encode_with_torch_fidelity_encoder(images, encoder, *, batch_size, image_range):
    images = _normalize_images(images, image_range=image_range)
    features = []
    for start in range(0, images.shape[0], batch_size):
        features.append(encoder.encode(images[start : start + batch_size]))
    return torch.cat(features, dim=0)


def _encode_with_torch_fidelity(images, spec, *, batch_size, image_range, device):
    encoder = _TorchFidelityFeatureEncoder(spec, device)
    features = _encode_with_torch_fidelity_encoder(
        images,
        encoder,
        batch_size=batch_size,
        image_range=image_range,
    )
    return features, encoder.extractor_name, encoder.feature_layer


def _metric_vectors(samples, reference, *, encoder, batch_size, image_range, device):
    if encoder is None:
        torch_device = _device_from_arg(device)
        sample_vectors = _normalize_images(samples, image_range=image_range).flatten(1).to(torch_device)
        reference_vectors = _normalize_images(reference, image_range=image_range).flatten(1).to(torch_device)
        return sample_vectors, reference_vectors, "pixels", "none"

    if isinstance(encoder, str):
        torch_fidelity_encoder = _TorchFidelityFeatureEncoder(encoder, device)
        sample_vectors = _encode_with_torch_fidelity_encoder(
            samples,
            torch_fidelity_encoder,
            batch_size=batch_size,
            image_range=image_range,
        )
        reference_vectors = _encode_with_torch_fidelity_encoder(
            reference,
            torch_fidelity_encoder,
            batch_size=batch_size,
            image_range=image_range,
        )
        return (
            sample_vectors,
            reference_vectors,
            "features",
            "torch_fidelity:{}:{}".format(
                torch_fidelity_encoder.extractor_name,
                torch_fidelity_encoder.feature_layer,
            ),
        )

    if not callable(encoder):
        raise TypeError("encoder must be None, a callable, or a torch_fidelity:* spec string.")

    sample_vectors = _encode_with_callable(
        samples,
        encoder,
        batch_size=batch_size,
        image_range=image_range,
        device=device,
    )
    reference_vectors = _encode_with_callable(
        reference,
        encoder,
        batch_size=batch_size,
        image_range=image_range,
        device=device,
    )
    encoder_name = getattr(encoder, "name", encoder.__class__.__name__)
    return sample_vectors, reference_vectors, "features", str(encoder_name)


def compute_f_mem_by_threshold(nn_ratio, thresholds):
    """Compute `f_mem` for a sweep of nearest-neighbor ratio thresholds."""
    ratios = torch.as_tensor(nn_ratio, dtype=torch.float32)
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]
    return {
        float(threshold): compute_f_mem(ratios, threshold=float(threshold))
        for threshold in thresholds
    }


def compute_image_memorization_metrics(
    samples,
    reference,
    *,
    encoder=None,
    ratio_threshold=0.5,
    query_batch_size=256,
    reference_batch_size=None,
    image_range="auto",
    device="cpu",
):
    """
    Compute real-image memorization metrics in pixel or feature space.
    """
    query_batch_size = int(query_batch_size)
    if query_batch_size < 1:
        raise ValueError(f"query_batch_size must be at least 1, got {query_batch_size}")
    sample_vectors, reference_vectors, space, encoder_name = _metric_vectors(
        samples,
        reference,
        encoder=encoder,
        batch_size=query_batch_size,
        image_range=image_range,
        device=device,
    )
    if sample_vectors.shape[1] != reference_vectors.shape[1]:
        raise ValueError(
            "Sample and reference vectors must have matching feature dimensions, "
            f"got {sample_vectors.shape[1]} and {reference_vectors.shape[1]}."
        )

    metrics = compute_memorization_metrics(
        sample_vectors,
        reference_vectors,
        ratio_threshold=ratio_threshold,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
    )
    metrics.update(
        {
            "space": space,
            "encoder": encoder_name,
            "image_range": str(image_range),
            "query_batch_size": query_batch_size,
            "reference_batch_size": (
                None if reference_batch_size is None else int(reference_batch_size)
            ),
            "feature_dim": int(sample_vectors.shape[1]),
        }
    )
    return metrics


def compute_image_folder_memorization_metrics(
    samples_path,
    reference_path,
    *,
    encoder=None,
    ratio_threshold=0.5,
    query_batch_size=256,
    reference_batch_size=None,
    image_range="auto",
    device="cpu",
):
    """Load two image/tensor folders and compute image memorization metrics."""
    samples = _load_collection(samples_path)
    reference = _load_collection(reference_path)
    metrics = compute_image_memorization_metrics(
        samples,
        reference,
        encoder=encoder,
        ratio_threshold=ratio_threshold,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
        image_range=image_range,
        device=device,
    )
    metrics["samples_path"] = str(Path(samples_path))
    metrics["reference_path"] = str(Path(reference_path))
    return metrics


def compute_image_fid(
    samples,
    reference=None,
    *,
    reference_stats_path=None,
    batch_size=64,
    device="cuda",
    image_range="auto",
    feature_extractor=DEFAULT_FID_FEATURE_EXTRACTOR,
    cache=False,
    verbose=False,
):
    """
    Compute FID from in-memory tensors using torch-fidelity.
    """
    if reference is None and reference_stats_path is None:
        raise ValueError("Pass either reference or reference_stats_path for FID.")
    if reference is not None and reference_stats_path is not None:
        raise ValueError("Pass only one of reference or reference_stats_path for FID.")

    try:
        import torch_fidelity
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch-fidelity is required for FID metrics.") from exc

    _select_cuda_device(device)
    samples_uint8 = _to_uint8_rgb(samples, image_range=image_range)
    input1 = _TensorImageDataset(samples_uint8)
    input2 = None
    reference_source = None
    n_reference = None

    kwargs = {
        "input1": input1,
        "input2": None,
        "fid": True,
        "cuda": _uses_cuda(device),
        "batch_size": int(batch_size),
        "feature_extractor": str(feature_extractor),
        "cache": bool(cache),
        "verbose": bool(verbose),
    }

    if reference is not None:
        reference_uint8 = _to_uint8_rgb(reference, image_range=image_range)
        input2 = _TensorImageDataset(reference_uint8)
        kwargs["input2"] = input2
        reference_source = "tensor"
        n_reference = len(input2)
    else:
        # The vendored torch-fidelity path reads fid_statistics_file only when
        # another metric prevents the FID-only shortcut. We compute and ignore
        # ISC so reference-stat FID can still use calculate_metrics directly.
        kwargs["fid_statistics_file"] = str(reference_stats_path)
        kwargs["isc"] = True
        reference_source = f"stats:{reference_stats_path}"

    raw_metrics = torch_fidelity.calculate_metrics(**kwargs)
    fid = float(raw_metrics["frechet_inception_distance"])
    return {
        "fid": fid,
        "frechet_inception_distance": fid,
        "n_samples": len(input1),
        "n_reference": n_reference,
        "feature_extractor": str(feature_extractor),
        "reference_source": reference_source,
    }


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
    """Load image/tensor folders and compute FID via torch-fidelity."""
    samples = _load_collection(samples_path)
    reference = _load_collection(reference_path) if reference_path is not None else None
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
    "DEFAULT_FEATURE_ENCODER",
    "compute_f_mem_by_threshold",
    "compute_image_fid",
    "compute_image_folder_fid",
    "compute_image_folder_memorization_metrics",
    "compute_image_memorization_metrics",
]
