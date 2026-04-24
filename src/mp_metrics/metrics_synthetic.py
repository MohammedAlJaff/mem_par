"""
Synthetic-data metrics.

This module uses the geometry saved by the synthetic data loaders. The main
distinction is:

1. Ambient metrics inspect the emitted vectors that the model trains on.
2. Intrinsic metrics project those vectors into known coordinates from the data
   generator, such as the GMM subspace or torus angle features.

The current metric set is deliberately small: nearest-neighbor ratio in ambient
space, nearest-neighbor ratio in intrinsic space, intrinsic sliced-W2, intrinsic
GMM negative log-likelihood when the dataset is a GMM, and torus-specific
coordinate summaries.
"""

import math

import torch


def _flatten_batch(tensor):
    """Convert a batch tensor to `[B, D]` for distance computation."""
    if tensor.dim() == 2:
        return tensor
    return tensor.flatten(1)


def _resolve_ratio_threshold(ratio_threshold=None, config=None):
    """Resolve the nearest-neighbor ratio threshold."""
    if ratio_threshold is not None:
        return float(ratio_threshold)
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

    The d1/d2 ratio is the main local-copying statistic. Values near zero mean
    the query has one much closer reference neighbor than the second neighbor.
    """
    n_reference = int(reference.shape[0])
    reference_batch_size = _resolve_batch_size(
        reference_batch_size,
        "reference_batch_size",
    )
    if reference_batch_size is None or reference_batch_size >= n_reference:
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

    for ref_start in range(0, n_reference, reference_batch_size):
        ref_end = min(ref_start + reference_batch_size, n_reference)
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
    Compute nearest-neighbor distances, indices, and d1/d2 ratios.

    This helper is used for both ambient vectors and intrinsic coordinates. It
    keeps the raw tensors so downstream code can report scalar summaries or
    inspect the nearest training examples directly.
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
    ratio = d1 / d2.clamp(min=1.0e-12)
    return {
        "d1": d1,
        "d2": d2,
        "r": ratio,
        "nn_ratio": ratio,
        "ratios": ratio,
        "nn_idx": nn_idx,
        "indices": nn_idx,
    }


def compute_f_mem(r, threshold=None, config=None):
    """
    Compute the fraction of samples below a nearest-neighbor ratio threshold.

    This scalar is retained for compatibility with older summaries, but the
    preferred synthetic metric names report the mean d1/d2 ratio directly.
    """
    threshold = _resolve_ratio_threshold(threshold, config=config)
    return float((r < threshold).float().mean().item())


def compute_memorization_metrics(
    samples,
    reference,
    ratio_threshold=None,
    chunk_size=256,
    query_batch_size=None,
    reference_batch_size=None,
    config=None,
):
    """
    Summarize nearest-neighbor ratios between samples and a reference set.

    For synthetic checkpoint evaluation, this generic helper is wrapped by
    `memorization_metrics_in_spaces`, which records separate ambient and
    intrinsic summaries.
    """
    threshold = _resolve_ratio_threshold(ratio_threshold, config=config)
    stats = compute_nn_stats(
        samples,
        reference,
        chunk_size=chunk_size,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
    )
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


def project_to_subspace(x, basis):
    """
    Project ambient samples into known GMM intrinsic coordinates.

    `basis` is orthonormal with shape `[D, d]`. Multiplying by `basis` maps from
    ambient `D` to intrinsic `d`; multiplying back by `basis.T` reconstructs the
    nearest point in the planted linear subspace.
    """
    intrinsic = x @ basis
    projected = intrinsic @ basis.T
    return {
        "intrinsic": intrinsic,
        "projected": projected,
    }


def gmm_intrinsic_log_prob(intrinsic, means, variance):
    """
    Evaluate the known equal-weight GMM density in intrinsic coordinates.

    This is only defined for `gmm_subspace`, where the generator stores the
    mixture means. It measures how plausible generated intrinsic coordinates
    are under the planted distribution.
    """
    if variance <= 0.0:
        raise ValueError("Intrinsic GMM variance must be positive.")

    n_components = int(means.shape[0])
    intrinsic_dim = int(means.shape[1])
    intrinsic_delta = intrinsic[:, None, :] - means[None, :, :]
    intrinsic_sq = intrinsic_delta.pow(2).sum(dim=2)
    log_components = (
        -0.5
        * (intrinsic_sq / variance + intrinsic_dim * math.log(2.0 * math.pi * variance))
        - math.log(float(n_components))
    )
    return torch.logsumexp(log_components, dim=1)


def intrinsic_sliced_w2(samples_intrinsic, reference_intrinsic, directions):
    """
    Compare two intrinsic distributions with random 1D Wasserstein projections.

    The same fixed directions are reused across checkpoints so a curve over
    training reflects model changes rather than new projection noise.
    """
    directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)
    sample_proj = samples_intrinsic @ directions.T
    reference_proj = reference_intrinsic @ directions.T
    sample_sorted, _ = sample_proj.sort(dim=0)
    reference_sorted, _ = reference_proj.sort(dim=0)
    count = min(sample_sorted.shape[0], reference_sorted.shape[0])
    return float((sample_sorted[:count] - reference_sorted[:count]).pow(2).mean().item())


def _geometry_kind(geometry, config=None):
    """Resolve the controlled synthetic geometry kind."""
    if geometry is None:
        if config is None:
            return ""
        return str(getattr(config.data, "dataset", ""))
    kind = geometry.get("kind") if isinstance(geometry, dict) else None
    if kind is not None:
        return str(kind)
    if "basis" in geometry:
        return "gmm_subspace"
    if config is not None:
        return str(getattr(config.data, "dataset", ""))
    return ""


def intrinsic_geometry_parts(samples, geometry, config=None):
    """
    Return intrinsic coordinates and geometry residuals for a controlled dataset.

    For GMM data this is a linear projection onto the planted subspace. For
    torus data this maps ambient samples back to the canonical 3D torus and
    then to periodic angle features `[cos(theta), sin(theta), cos(phi), sin(phi)]`.
    """
    kind = _geometry_kind(geometry, config)
    if kind == "gmm_subspace":
        parts = project_to_subspace(samples, geometry["basis"])
        return {
            "kind": kind,
            "intrinsic": parts["intrinsic"],
            "geometry_error": (samples - parts["projected"]).norm(dim=1),
            "projected": parts["projected"],
        }
    if kind == "torus3d":
        from mp_data.torus import (
            torus_angles_to_xyz,
            torus_periodic_features,
            torus_xyz_to_angles,
        )

        major_radius = float(geometry["major_radius"])
        minor_radius = float(geometry["minor_radius"])
        basis = geometry.get("basis")
        torus_xyz = samples @ basis if basis is not None else samples
        theta, phi = torus_xyz_to_angles(torus_xyz, major_radius, minor_radius)
        intrinsic = torus_periodic_features(theta, phi)
        nearest_torus_xyz = torus_angles_to_xyz(theta, phi, major_radius, minor_radius)
        if basis is not None:
            projected = nearest_torus_xyz @ basis.T
        else:
            projected = nearest_torus_xyz
        geometry_error = torch.linalg.norm(torus_xyz - nearest_torus_xyz, dim=1)
        return {
            "kind": kind,
            "intrinsic": intrinsic,
            "geometry_error": geometry_error,
            "projected": projected,
            "torus_xyz": torus_xyz,
            "nearest_torus_xyz": nearest_torus_xyz,
            "theta": theta,
            "phi": phi,
        }
    raise ValueError(f"Unsupported controlled synthetic geometry: {kind!r}")


def memorization_metrics_in_spaces(samples, train_x, geometry, train_intrinsic, *, chunk_size, config):
    """
    Run nearest-neighbor ratio checks in ambient and intrinsic spaces.

    The ambient ratio asks whether generated emitted vectors copy train vectors.
    The intrinsic ratio asks the same question after projecting into the known
    coordinates of the synthetic generator.
    """
    ambient = compute_nn_stats(
        samples,
        train_x,
        chunk_size=chunk_size,
    )
    intrinsic = intrinsic_geometry_parts(samples, geometry, config=config)["intrinsic"]
    intrinsic_metrics = compute_nn_stats(
        intrinsic,
        train_intrinsic,
        chunk_size=chunk_size,
    )
    return {
        "ambient_nearest_neighbor_ratio_d1_over_d2": float(ambient["r"].mean().item()),
        "intrinsic_nearest_neighbor_ratio_d1_over_d2": float(
            intrinsic_metrics["r"].mean().item()
        ),
    }


def gmm_intrinsic_metrics(samples, basis, means, sigma_z, sigma_eta, reference_intrinsic, directions):
    """
    Measure generated samples in planted GMM intrinsic coordinates.

    The returned NLL uses the known mixture density. The sliced-W2 term compares
    generated intrinsic coordinates against the saved reference intrinsic
    coordinates.
    """
    geometry = {"kind": "gmm_subspace", "basis": basis, "means": means}
    parts = intrinsic_geometry_parts(samples, geometry)
    intrinsic = parts["intrinsic"]
    intrinsic_var = float(sigma_z) ** 2 + float(sigma_eta) ** 2
    log_density = gmm_intrinsic_log_prob(intrinsic, means, intrinsic_var)
    intrinsic_nll_values = -log_density

    return {
        "intrinsic_gmm_nll_mean": float(intrinsic_nll_values.mean().item()),
        "intrinsic_sliced_w2": intrinsic_sliced_w2(intrinsic, reference_intrinsic, directions),
    }


def torus3d_metrics(samples, geometry, reference_intrinsic, directions):
    """
    Measure generated samples against the known torus intrinsic coordinates.

    Torus evaluation uses periodic angle features rather than raw angles so the
    metric is continuous across the `-pi`/`pi` wrap point.
    """
    parts = intrinsic_geometry_parts(samples, geometry)
    return {
        "intrinsic_sliced_w2": intrinsic_sliced_w2(parts["intrinsic"], reference_intrinsic, directions),
    }


def synthetic_distribution_metrics(samples, geometry, reference_intrinsic, directions, config):
    """
    Dispatch dataset-specific intrinsic distribution metrics.

    This function is intentionally explicit: GMM gets intrinsic NLL plus
    sliced-W2, while torus gets sliced-W2 on periodic angle features.
    """
    kind = _geometry_kind(geometry, config)
    if kind == "gmm_subspace":
        return gmm_intrinsic_metrics(
            samples,
            geometry["basis"],
            geometry["means"],
            float(config.data.sigma_z),
            float(config.data.sigma_eta),
            reference_intrinsic,
            directions,
        )
    if kind == "torus3d":
        return torus3d_metrics(samples, geometry, reference_intrinsic, directions)
    raise ValueError(f"Unsupported controlled synthetic geometry: {kind!r}")


def collect_controlled_synthetic_metric_record(
    *,
    step,
    samples,
    baseline_samples,
    train_tensor,
    train_intrinsic,
    geometry,
    reference_intrinsic,
    sliced_directions,
    chunk_size,
    config,
):
    """
    Collect generated and held-out synthetic diagnostics into one flat record.

    Both generated samples and held-out reference samples are compared against
    the saved training tensor. The held-out record is the baseline for what
    non-training samples look like under the same metrics.
    """
    generated_mem = memorization_metrics_in_spaces(
        samples,
        train_tensor,
        geometry,
        train_intrinsic,
        chunk_size=chunk_size,
        config=config,
    )
    generated_distribution = synthetic_distribution_metrics(
        samples,
        geometry,
        reference_intrinsic,
        sliced_directions,
        config,
    )
    holdout_mem = memorization_metrics_in_spaces(
        baseline_samples,
        train_tensor,
        geometry,
        train_intrinsic,
        chunk_size=chunk_size,
        config=config,
    )
    holdout_intrinsic = intrinsic_geometry_parts(
        baseline_samples,
        geometry,
        config=config,
    )["intrinsic"]
    holdout_distribution = synthetic_distribution_metrics(
        baseline_samples,
        geometry,
        holdout_intrinsic,
        sliced_directions,
        config,
    )

    record = {"step": int(step)}
    record.update(generated_mem)
    record.update(generated_distribution)
    record.update({"holdout_{}".format(key): value for key, value in holdout_mem.items()})
    record.update({"holdout_{}".format(key): value for key, value in holdout_distribution.items()})
    return record



__all__ = [
    "compute_f_mem",
    "compute_memorization_metrics",
    "compute_nn_stats",
    "collect_controlled_synthetic_metric_record",
    "gmm_intrinsic_log_prob",
    "gmm_intrinsic_metrics",
    "intrinsic_geometry_parts",
    "intrinsic_sliced_w2",
    "memorization_metrics_in_spaces",
    "project_to_subspace",
    "synthetic_distribution_metrics",
    "torus3d_metrics",
]


if __name__ == "__main__":
    print("mp_metrics.metrics_synthetic example")
    reference = torch.randn(8, 2)
    samples = reference[:3] + 0.05 * torch.randn(3, 2)
    metrics = compute_memorization_metrics(samples, reference, ratio_threshold=0.5)
    print(f"mean_nn_ratio: {metrics['mean_nn_ratio']:.6f}")
