"""
Image metrics for generated samples and reference images.

Read this module as the image analogue of `mp_metrics.synthetic`:

1. Nearest-neighbor metrics compare generated images to the saved training
   tensor. They report first-neighbor distance, second-neighbor distance, and
   their ratio. A low ratio means a generated image is much closer to one train
   image than to its next nearest train image.
2. Image metrics can run either in pixel space or in feature space. Feature
   space uses a batched encoder, usually torch-fidelity's Inception features.
3. FID is kept in this file because it has image-specific normalization and
   torch-fidelity input requirements.

All image tensors are normalized to `[0, 1]` before metric extraction. The
training loop stores images in the model range, usually `[-1, 1]`; metrics and
figures convert that representation at the boundary.
"""

import warnings

import torch
from torch.utils.data import Dataset

DEFAULT_FID_FEATURE_EXTRACTOR = "inception-v3-compat"
RHO_SWEEP = (0.3, 0.5, 0.7)


def _flatten_batch(tensor):
    """Convert a batch tensor to `[B, D]` for distance computation."""
    if tensor.dim() == 2:
        return tensor
    return tensor.flatten(1)


def _resolve_rho(rho=None, config=None):
    """Resolve rho, the nearest-neighbor ratio threshold."""
    if rho is not None:
        return float(rho)
    if config is not None and hasattr(config, "evaluation"):
        evaluation = config.evaluation
        if hasattr(evaluation, "ratio_threshold"):
            return float(evaluation.ratio_threshold)
    return 0.5


def _resolve_batch_size(value, name):
    """Resolve an optional positive batch size."""
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _top2_distances(query, reference, reference_batch_size=None):
    """
    Compute the closest and second-closest reference distances for each query.

    The d1/d2 ratio is the main memorization signal. Keeping both distances
    also makes it possible to diagnose whether a low ratio comes from one very
    close neighbor or from generally dense reference data.
    """
    N = int(reference.shape[0])
    reference_batch_size = _resolve_batch_size(
        reference_batch_size,
        "reference_batch_size",
    )
    if reference_batch_size is None or reference_batch_size >= N:
        distances = torch.cdist(query, reference)
        topk = torch.topk(distances, k=2, largest=False, dim=1)
        return topk.values[:, 0], topk.values[:, 1], topk.indices[:, 0]

    best_values = torch.full(
        (query.shape[0], 2),
        float("inf"),
        dtype=query.dtype,
        device=query.device,
    )
    best_indices = torch.full(
        (query.shape[0], 2),
        -1,
        dtype=torch.long,
        device=query.device,
    )
    for ref_start in range(0, N, reference_batch_size):
        ref_end = min(ref_start + reference_batch_size, N)
        ref_chunk = reference[ref_start:ref_end]
        distances = torch.cdist(query, ref_chunk)
        local_k = min(2, int(ref_chunk.shape[0]))
        local_topk = torch.topk(distances, k=local_k, largest=False, dim=1)
        local_indices = local_topk.indices + ref_start

        candidate_values = torch.cat([best_values, local_topk.values], dim=1)
        candidate_indices = torch.cat([best_indices, local_indices], dim=1)
        merged = torch.topk(candidate_values, k=2, largest=False, dim=1)
        best_values = merged.values
        best_indices = torch.gather(candidate_indices, 1, merged.indices)

    return best_values[:, 0], best_values[:, 1], best_indices[:, 0]


def compute_nn_stats(
    samples,
    reference,
    chunk_size=256,
    query_batch_size=None,
    reference_batch_size=None,
):
    """
    Compute nearest-neighbor statistics for a batch of metric vectors.

    Inputs may be images or already-flat vectors. The output keeps tensor-level
    distances and `nn_idx` so callers can either summarize them or build
    preview figures that point back to the nearest training examples.
    """
    query_batch_size = _resolve_batch_size(
        chunk_size if query_batch_size is None else query_batch_size,
        "query_batch_size",
    )
    samples = _flatten_batch(samples).float()
    reference = _flatten_batch(reference).float()
    if reference.shape[0] < 2:
        raise ValueError("Nearest-neighbor metrics require at least two reference samples.")

    d1_list = []
    d2_list = []
    i1_list = []
    for start in range(0, samples.shape[0], query_batch_size):
        chunk = samples[start : start + query_batch_size]
        d1, d2, nn_idx = _top2_distances(
            chunk,
            reference,
            reference_batch_size=reference_batch_size,
        )
        d1_list.append(d1)
        d2_list.append(d2)
        i1_list.append(nn_idx)

    d1 = torch.cat(d1_list)
    d2 = torch.cat(d2_list)
    nn_idx = torch.cat(i1_list)
    r = d1 / d2.clamp(min=1.0e-12)  # r(x) = d1(x) / d2(x), NN ratio
    return {
        "d1": d1,
        "d2": d2,
        "r": r,
        "nn_idx": nn_idx,
    }


def compute_f_mem(r, rho=None, config=None):
    """
    Compute the fraction of samples below rho.

    This is a scalar summary of the d1/d2 ratios. It is not a proof of
    memorization by itself; it is a screening statistic for unusually close
    train-neighbor matches.
    """
    rho = _resolve_rho(rho, config=config)
    f_mem = (r < rho).float().mean()  # f_mem(rho) = fraction with r(x) < rho
    return float(f_mem.item())


def _memorization_summary_from_stats(stats, *, rho, n_samples, N):
    """
    Summarize nearest-neighbor statistics with paper notation.

    `N` is the training/reference set size used for nearest-neighbor search.
    """
    r = stats["r"]  # r(x) = d1(x) / d2(x), NN ratio
    return {
        "rho": float(rho),
        "n_samples": int(n_samples),
        "N": int(N),
        "f_mem": compute_f_mem(r, rho=rho),
        "mean_r": float(r.mean().item()),
        "median_r": float(r.median().item()),
        "mean_d1": float(stats["d1"].mean().item()),
        "mean_d2": float(stats["d2"].mean().item()),
        "median_d1": float(stats["d1"].median().item()),
        "median_d2": float(stats["d2"].median().item()),
    }


def compute_memorization_metrics(
    samples,
    reference,
    rho=None,
    chunk_size=256,
    query_batch_size=None,
    reference_batch_size=None,
    config=None,
):
    """
    Summarize nearest-neighbor ratios for generated samples against reference data.

    The caller chooses what `reference` means. In checkpoint evaluation it is
    the saved training tensor; for baseline checks it can be held-out reference
    images compared against the same training tensor.
    """
    rho = _resolve_rho(rho, config=config)
    stats = compute_nn_stats(
        samples,
        reference,
        chunk_size=chunk_size,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
    )
    return _memorization_summary_from_stats(
        stats,
        rho=rho,
        n_samples=samples.shape[0],
        N=reference.shape[0],
    )


class _TensorImageDataset(Dataset):
    """
    Torch-fidelity adapter for an in-memory uint8 RGB image batch.

    torch-fidelity expects a Dataset-like object. This wrapper lets us evaluate
    generated tensors without first writing them as image files.
    """

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

    The runner stores model inputs in a training-friendly range. Metric code
    uses display/encoder-friendly pixels, so this is the main range boundary.
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
    warnings.warn(
        "image_range='auto' inferred range from data "
        f"(min={min_value:.4g}, max={max_value:.4g}). "
        "Set image_range explicitly for reproducibility.",
        stacklevel=2,
    )
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

    FID does not consume normalized model tensors directly. This conversion is
    deliberately explicit so grayscale and alpha-channel inputs are handled in
    one place.
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
    """Convert a device argument into `torch.device`."""
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _uses_cuda(device):
    """Return whether a device argument selects CUDA."""
    return _device_from_arg(device).type == "cuda"


def _select_cuda_device(device):
    """Select an indexed CUDA device before torch-fidelity calls."""
    torch_device = _device_from_arg(device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device.index)
    return torch_device


def _coerce_feature_output(output):
    """
    Convert an encoder output to a `[B, F]` feature matrix.

    Encoders differ in whether they return tensors, tuples, or dictionaries.
    Metric code should only see one flat feature vector per image.
    """
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
    """
    Encode images with a user-provided callable while preserving its mode.

    The encoder may be a research model passed from a notebook. We temporarily
    put it in eval mode, run without gradients, and restore its previous
    training state before returning.
    """
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
    """
    Batched feature extractor wrapper for torch-fidelity encoders.

    This keeps torch-fidelity setup separate from the metric math. The public
    metric functions receive only feature matrices after this wrapper runs.
    """

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
    """Parse a torch-fidelity encoder specification string."""
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
    """Encode images with a constructed torch-fidelity encoder."""
    images = _normalize_images(images, image_range=image_range)
    features = []
    for start in range(0, images.shape[0], batch_size):
        features.append(encoder.encode(images[start : start + batch_size]))
    return torch.cat(features, dim=0)


def _encode_with_torch_fidelity(images, spec, *, batch_size, image_range, device):
    """Construct and run a torch-fidelity encoder."""
    encoder = _TorchFidelityFeatureEncoder(spec, device)
    features = _encode_with_torch_fidelity_encoder(
        images,
        encoder,
        batch_size=batch_size,
        image_range=image_range,
    )
    return features, encoder.extractor_name, encoder.feature_layer


def _metric_vectors(samples, reference, *, encoder, batch_size, image_range, device):
    """
    Convert image batches to the vector space used by nearest-neighbor metrics.

    With `encoder=None`, the metric space is raw normalized pixels. With an
    encoder, both sample and reference images are embedded with the same feature
    extractor before distances are computed.
    """
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


def compute_f_mem_by_rho(r, rho_values):
    """
    Compute `f_mem` for a sweep of rho values.

    This is useful when checking whether a conclusion depends on one arbitrary
    cutoff value.
    """
    r = torch.as_tensor(r, dtype=torch.float32)
    if isinstance(rho_values, (int, float)):
        rho_values = [rho_values]
    return {
        float(rho): compute_f_mem(r, rho=float(rho))
        for rho in rho_values
    }


def compute_image_memorization_metrics(
    samples,
    reference,
    *,
    encoder=None,
    rho=None,
    query_batch_size=256,
    reference_batch_size=None,
    image_range="auto",
    device="cpu",
):
    """
    Compute image nearest-neighbor memorization metrics.

    `samples` are usually model generations. `reference` is usually the saved
    training tensor. The returned record states whether distances were measured
    in pixel space or encoder feature space.
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
        rho=rho,
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

    The function accepts either a reference tensor or a saved reference-stat
    file. It converts generated images to uint8 RGB because that is the
    torch-fidelity input contract.
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


def collect_image32_metric_record(
    step,
    samples,
    baseline_samples,
    train_tensor,
    *,
    chunk_size,
    config,
):
    """
    Collect generated and held-out image nearest-neighbor metrics.

    Generated samples and held-out reference images are both compared against
    the saved training tensor. The held-out comparison is the baseline: it shows
    what nearest-neighbor ratios look like for real images not used in training.
    """
    rho = _resolve_rho(config=config)
    baseline_samples = baseline_samples[: samples.shape[0]]
    generated_stats = compute_nn_stats(
        samples,
        train_tensor,
        chunk_size=int(chunk_size),
    )
    baseline_stats = compute_nn_stats(
        baseline_samples,
        train_tensor,
        chunk_size=int(chunk_size),
    )
    generated = _memorization_summary_from_stats(
        generated_stats,
        rho=rho,
        n_samples=samples.shape[0],
        N=train_tensor.shape[0],
    )
    baseline = _memorization_summary_from_stats(
        baseline_stats,
        rho=rho,
        n_samples=baseline_samples.shape[0],
        N=train_tensor.shape[0],
    )

    record = {
        "step": int(step),
        "rho": generated["rho"],
        "f_mem": generated["f_mem"],
        "baseline_f_mem": baseline["f_mem"],
        "delta_f_mem": generated["f_mem"] - baseline["f_mem"],  # Delta f_mem = gen - held-out
        "mean_r": generated["mean_r"],
        "median_r": generated["median_r"],
        "baseline_mean_r": baseline["mean_r"],
        "baseline_median_r": baseline["median_r"],
        "mean_d1": generated["mean_d1"],
        "mean_d2": generated["mean_d2"],
        "baseline_mean_d1": baseline["mean_d1"],
        "baseline_mean_d2": baseline["mean_d2"],
    }
    for rho_val in RHO_SWEEP:
        f_gen = compute_f_mem(generated_stats["r"], rho=rho_val)  # f_mem(rho) for generated samples
        f_base = compute_f_mem(baseline_stats["r"], rho=rho_val)  # f_mem(rho) for held-out baseline
        record[f"f_mem_rho{rho_val}"] = f_gen
        record[f"baseline_f_mem_rho{rho_val}"] = f_base
        record[f"delta_f_mem_rho{rho_val}"] = f_gen - f_base  # Delta f_mem(rho)
    return record


__all__ = [
    "DEFAULT_FID_FEATURE_EXTRACTOR",
    "RHO_SWEEP",
    "collect_image32_metric_record",
    "compute_f_mem_by_rho",
    "compute_f_mem",
    "compute_image_fid",
    "compute_image_memorization_metrics",
    "compute_memorization_metrics",
    "compute_nn_stats",
]


if __name__ == "__main__":
    print("mp_metrics.images example")
    reference = torch.rand(4, 1, 4, 4)
    samples = reference[:2].clone()
    metrics = compute_image_memorization_metrics(samples, reference, image_range="0_1")
    print(f"mean_r: {metrics['mean_r']:.6f}")
