"""
Synthetic Gaussian-mixture data on a linear subspace.

This module builds the controlled high-dimensional synthetic dataset used to
study how memorization changes with ambient dimension, intrinsic dimension, and
subspace geometry. The returned bundle includes both the sampled tensors and
the fixed geometry needed to replay the same study later.
"""

import torch


def _make_geometry(config):
    """
    Sample the fixed subspace basis and cluster means for one study geometry.

    We separate geometry from sample draws on purpose: the geometry seed should
    control the underlying manifold, while the training seed controls which
    specific samples were drawn from that manifold.
    """
    data_dim = int(config.data.data_dim)
    intrinsic_dim = int(config.data.intrinsic_dim)
    n_clusters = int(config.data.n_clusters)

    generator = torch.Generator().manual_seed(int(config.data.geometry_seed))
    basis, _ = torch.linalg.qr(torch.randn(data_dim, intrinsic_dim, generator=generator))
    means = torch.randn(n_clusters, intrinsic_dim, generator=generator)

    return basis[:, :intrinsic_dim], means


def _sample_gmm(config, basis, means, n_samples, seed):
    """
    Draw ambient-space samples from the configured latent Gaussian mixture.
    """
    generator = torch.Generator().manual_seed(int(seed))
    cluster_ids = torch.randint(
        0, means.shape[0], (n_samples,), generator=generator
    )
    latent_noise = config.data.sigma_z * torch.randn(
        n_samples, means.shape[1], generator=generator
    )
    latent = means[cluster_ids] + latent_noise
    ambient = latent @ basis.T
    ambient = ambient + config.data.sigma_eta * torch.randn(
        n_samples, basis.shape[0], generator=generator
    )
    return ambient.contiguous(), cluster_ids


def load_gmm_subspace_bundle(config):
    """
    Build the canonical train/reference bundle for the GMM subspace study.

    The training and reference sets share the same geometry but use different
    sampling seeds so evaluation is not just replaying the exact same points.
    Saving that geometry in the returned bundle is what later lets us explain
    whether a memorization effect came from the ambient samples alone or from
    the underlying subspace structure.
    """
    basis, means = _make_geometry(config)
    train_tensor, train_labels = _sample_gmm(
        config,
        basis,
        means,
        int(config.data.n_train),
        int(config.training.seed),
    )
    reference_tensor, reference_labels = _sample_gmm(
        config,
        basis,
        means,
        int(config.data.n_eval),
        int(config.training.seed) + 1,
    )

    return {
        "train": train_tensor,
        "train_labels": train_labels,
        "reference": reference_tensor,
        "reference_labels": reference_labels,
        "train_role": "train",
        "reference_role": "heldout_reference",
        "train_source_id": f"gmm_subspace:seed:{int(config.training.seed)}",
        "reference_source_id": f"gmm_subspace:seed:{int(config.training.seed) + 1}",
        "train_split": "synthetic_train_seed",
        "reference_split": "synthetic_reference_seed",
        "geometry": {
            "basis": basis,
            "means": means,
        },
        "display_shape": None,
    }


if __name__ == "__main__":
    from mp_config import get_base_config

    print("mp_data.gmm example")

    # This is the fully runnable synthetic-data loader example.
    config = get_base_config()
    config.data.dataset = "gmm_subspace"
    config.data.n_train = 8
    config.data.n_eval = 6
    config.data.data_dim = 8
    config.data.intrinsic_dim = 2
    config.data.n_clusters = 3

    # Build one bundle and inspect both the sampled tensors and the geometry.
    bundle = load_gmm_subspace_bundle(config)

    # These prints check the shared bundle contract plus the stored subspace data.
    print(f"train shape           : {tuple(bundle['train'].shape)}")
    print(f"reference shape       : {tuple(bundle['reference'].shape)}")
    print(f"train_labels shape    : {tuple(bundle['train_labels'].shape)}")
    print(f"basis shape           : {tuple(bundle['geometry']['basis'].shape)}")
    print(f"means shape           : {tuple(bundle['geometry']['means'].shape)}")
