import torch


def _make_geometry(config):
    data_dim = int(config.data.data_dim)
    intrinsic_dim = int(config.data.intrinsic_dim)
    n_clusters = int(config.data.n_clusters)

    generator = torch.Generator().manual_seed(int(config.data.geometry_seed))
    basis, _ = torch.linalg.qr(torch.randn(data_dim, intrinsic_dim, generator=generator))
    means = torch.randn(n_clusters, intrinsic_dim, generator=generator)

    return basis[:, :intrinsic_dim], means


def _sample_gmm(config, basis, means, n_samples, seed):
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
        "geometry": {
            "basis": basis,
            "means": means,
        },
        "display_shape": None,
    }
