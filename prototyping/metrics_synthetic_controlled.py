"""
Metrics for controlled synthetic prototype runs.

These helpers deliberately do not build data, train models, or sample from
models. They measure already-generated tensors against known synthetic geometry.
"""

import math


def project_to_subspace(x, basis):
    """
    Split ambient samples into true latent coordinates and off-subspace error.

    The planted data model is x = z B.T + noise, so x @ B is the coordinate
    system where the known GMM really lives. The residual tells us how much a
    generated sample has drifted away from that planted subspace.
    """
    latent = x @ basis
    projected = latent @ basis.T
    residual = x - projected
    residual_norm = residual.norm(dim=1)
    return {
        "latent": latent,
        "projected": projected,
        "residual": residual,
        "residual_norm": residual_norm,
    }


def gmm_latent_log_prob(latent, means, variance):
    """
    Evaluate the known equal-weight latent GMM and return responsibilities.

    This is the oracle density in intrinsic coordinates. A good generator should
    put samples near these means with the designed variance, not merely near the
    finite training set.
    """
    import torch

    if variance <= 0.0:
        raise ValueError("Latent GMM variance must be positive.")

    n_components = int(means.shape[0])
    intrinsic_dim = int(means.shape[1])
    latent_delta = latent[:, None, :] - means[None, :, :]
    latent_sq = latent_delta.pow(2).sum(dim=2)
    log_components = (
        -0.5 * (latent_sq / variance + intrinsic_dim * math.log(2.0 * math.pi * variance))
        - math.log(float(n_components))
    )
    log_density = torch.logsumexp(log_components, dim=1)
    responsibilities = (log_components - log_density[:, None]).exp()
    return log_density, responsibilities


def component_coverage_metrics(responsibilities):
    """
    Check whether generated samples cover the true mixture components.

    Likelihood can look acceptable while a model ignores one component. Averaged
    responsibilities expose that mode-balance failure directly.
    """
    n_components = int(responsibilities.shape[1])
    uniform_weight = 1.0 / float(n_components)
    component_probs = responsibilities.mean(dim=0)
    component_probs_safe = component_probs.clamp(min=1.0e-12)
    return {
        "component_weight_l1": float((component_probs - uniform_weight).abs().sum().item()),
        "component_weight_kl": float(
            (component_probs_safe * (component_probs_safe.log() - math.log(uniform_weight))).sum().item()
        ),
        "component_probs": [float(value) for value in component_probs.tolist()],
    }


def latent_frechet_distance(samples_latent, means, latent_var):
    """
    FID-style moment distance in the true latent coordinates.

    This compares generated mean/covariance against the analytic GMM mean and
    covariance. It is useful, but it is only a moment check: it can miss mode
    collapse when moments happen to match.
    """
    import torch

    intrinsic_dim = int(means.shape[1])
    oracle_mean = means.mean(dim=0)
    centered_means = means - oracle_mean
    oracle_cov = centered_means.T @ centered_means / float(means.shape[0])
    oracle_cov = oracle_cov + float(latent_var) * torch.eye(
        intrinsic_dim,
        dtype=means.dtype,
        device=means.device,
    )

    samples_centered = samples_latent - samples_latent.mean(dim=0, keepdim=True)
    samples_cov = samples_centered.T @ samples_centered / float(max(1, samples_latent.shape[0]))

    oracle_eigvals, oracle_eigvecs = torch.linalg.eigh(oracle_cov)
    oracle_sqrt = (
        oracle_eigvecs
        @ torch.diag(oracle_eigvals.clamp(min=0.0).sqrt())
        @ oracle_eigvecs.T
    )
    middle = oracle_sqrt @ samples_cov @ oracle_sqrt
    middle = 0.5 * (middle + middle.T)
    middle_eigvals = torch.linalg.eigvalsh(middle).clamp(min=0.0)
    frechet = (
        (samples_latent.mean(dim=0) - oracle_mean).pow(2).sum()
        + torch.trace(samples_cov)
        + torch.trace(oracle_cov)
        - 2.0 * middle_eigvals.sqrt().sum()
    )
    return float(max(0.0, frechet.item()))


def latent_sliced_w2(samples_latent, reference_latent, directions):
    """
    Compare latent distributions through many simple 1D Wasserstein checks.

    Each random direction turns the clouds into sorted scalar lists. Matching
    those sorted lists is a lightweight way to catch shape differences that a
    mean/covariance metric can miss.
    """
    directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)
    sample_proj = samples_latent @ directions.T
    reference_proj = reference_latent @ directions.T
    sample_sorted, _ = sample_proj.sort(dim=0)
    reference_sorted, _ = reference_proj.sort(dim=0)
    count = min(sample_sorted.shape[0], reference_sorted.shape[0])
    return float((sample_sorted[:count] - reference_sorted[:count]).pow(2).mean().item())


def oracle_gmm_metrics(samples, basis, means, sigma_z, sigma_eta, reference_latent, directions):
    """
    Measure generated samples against the distribution we planted.

    The latent NLL asks whether samples sit under the true GMM. The full NLL adds
    the ambient residual penalty when sigma_eta is positive. If sigma_eta is
    zero, the ambient density is singular, so full NLL is intentionally undefined.
    """
    import torch

    subspace = project_to_subspace(samples, basis)
    latent = subspace["latent"]
    residual = subspace["residual"]
    latent_var = float(sigma_z) ** 2 + float(sigma_eta) ** 2
    log_density, responsibilities = gmm_latent_log_prob(latent, means, latent_var)
    latent_nll_values = -log_density

    data_dim = int(samples.shape[1])
    intrinsic_dim = int(basis.shape[1])
    residual_dim = data_dim - intrinsic_dim
    residual_norm_sq = residual.pow(2).sum(dim=1)
    if sigma_eta > 0.0:
        if residual_dim > 0:
            residual_nll_values = 0.5 * (
                residual_norm_sq / (float(sigma_eta) ** 2)
                + residual_dim * math.log(2.0 * math.pi * float(sigma_eta) ** 2)
            )
        else:
            residual_nll_values = torch.zeros_like(latent_nll_values)
        full_nll_mean = float((latent_nll_values + residual_nll_values).mean().item())
    else:
        full_nll_mean = None

    coverage = component_coverage_metrics(responsibilities)
    total_energy = samples.pow(2).sum().clamp(min=1.0e-12)
    metrics = {
        "oracle_latent_nll_mean": float(latent_nll_values.mean().item()),
        "oracle_latent_nll_std": float(latent_nll_values.std(unbiased=False).item()),
        "oracle_full_nll_mean": full_nll_mean,
        "off_subspace_energy_fraction": float(residual.pow(2).sum().item() / total_energy.item()),
        "latent_frechet_distance_to_oracle": latent_frechet_distance(latent, means, latent_var),
        "latent_sliced_w2": latent_sliced_w2(latent, reference_latent, directions),
        "residual_norm_mean": float(subspace["residual_norm"].mean().item()),
    }
    metrics.update(coverage)
    return metrics


def memorization_metrics_in_spaces(samples, train_x, basis, train_latent, *, chunk_size, config):
    """
    Run nearest-neighbor memorization checks in ambient and true latent spaces.

    Ambient space is what the model emits. Latent space removes off-subspace noise
    and asks whether samples are copying the finite training set on the real GMM
    coordinates.
    """
    from mp_metrics import compute_memorization_metrics

    ambient = compute_memorization_metrics(
        samples,
        train_x,
        chunk_size=chunk_size,
        config=config,
    )
    latent = project_to_subspace(samples, basis)["latent"]
    latent_metrics = compute_memorization_metrics(
        latent,
        train_latent,
        chunk_size=chunk_size,
        config=config,
    )
    return {
        "ambient_f_mem": ambient["f_mem"],
        "ambient_mean_nn_ratio": ambient["mean_nn_ratio"],
        "ambient_std_nn_ratio": ambient["std_nn_ratio"],
        "latent_f_mem": latent_metrics["f_mem"],
        "latent_mean_nn_ratio": latent_metrics["mean_nn_ratio"],
        "latent_std_nn_ratio": latent_metrics["std_nn_ratio"],
    }


__all__ = [
    "component_coverage_metrics",
    "gmm_latent_log_prob",
    "latent_frechet_distance",
    "latent_sliced_w2",
    "memorization_metrics_in_spaces",
    "oracle_gmm_metrics",
    "project_to_subspace",
]
