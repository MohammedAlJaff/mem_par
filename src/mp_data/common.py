import torch


def seeded_indices(size, count, seed):
    if count > size:
        raise ValueError(f"Requested {count} items from a collection of size {size}")

    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(size, generator=generator)[:count]


def maybe_flatten(tensor, config):
    if getattr(config.data, "flatten", False):
        return tensor.flatten(1)
    return tensor


def build_data_bundle(config):
    dataset_name = config.data.dataset

    if dataset_name == "mnist":
        from mp_data.mnist import load_mnist_bundle

        return load_mnist_bundle(config)
    if dataset_name == "gmm_subspace":
        from mp_data.gmm import load_gmm_subspace_bundle

        return load_gmm_subspace_bundle(config)
    if dataset_name == "celeba32_grey":
        from mp_data.celeba32 import load_celeba32_bundle

        return load_celeba32_bundle(config)

    raise ValueError(f"Unknown dataset: {dataset_name}")
