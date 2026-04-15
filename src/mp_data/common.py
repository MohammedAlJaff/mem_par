"""
Shared data-bundle helpers for first-party studies.

This file contains the small helpers used by every dataset loader:
- deterministic subset selection from a larger dataset
- optional flattening when a study wants vectors instead of image tensors
- dataset-name dispatch into the correct loader

Canonical bundle contract:
    train:              the tensor seen by the optimizer
    reference:          the tensor used for evaluation comparisons
    train_labels:       optional labels aligned with `train`
    reference_labels:   optional labels aligned with `reference`
    train_indices:      optional source indices for deterministic replay
    reference_indices:  optional source indices for deterministic replay
    display_shape:      optional original image shape for plotting utilities
    geometry:           optional dataset-specific structure such as GMM basis
"""

import torch


def seeded_indices(size, count, seed):
    """
    Draw `count` unique indices from `range(size)` using a fixed seed.
    """
    if count > size:
        raise ValueError(f"Requested {count} items from a collection of size {size}")

    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(size, generator=generator)[:count]


def maybe_flatten(tensor, config):
    """
    Flatten image-like tensors to `[B, D]` when the study config asks for it.
    """
    if getattr(config.data, "flatten", False):
        return tensor.flatten(1)
    return tensor


def build_data_bundle(config):
    """
    Dispatch from `config.data.dataset` to the corresponding canonical loader.

    Every loader returns the same high-level bundle contract documented at the
    top of this module.
    """
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


if __name__ == "__main__":
    from mp_config import get_base_config

    torch.manual_seed(0)

    print("mp_data.common example")

    # Show the two small helper contracts first: seeded subset selection and
    # optional flattening of image-shaped tensors.
    indices = seeded_indices(size=10, count=4, seed=3)
    image_batch = torch.randn(2, 1, 4, 4)

    # Build a tiny synthetic-data config so `build_data_bundle(...)` is runnable.
    config = get_base_config()
    config.data.dataset = "gmm_subspace"
    config.data.n_train = 8
    config.data.n_eval = 6
    config.data.data_dim = 8
    config.data.intrinsic_dim = 2
    config.data.n_clusters = 3
    config.data.flatten = False

    # Materialize the helper outputs and then the full bundle contract.
    flattened = maybe_flatten(image_batch, config)
    bundle = build_data_bundle(config)

    # The key check here is that every loader returns the same bundle keys.
    print(f"seeded_indices        : {indices.tolist()}")
    print(f"maybe_flatten shape   : {tuple(flattened.shape)}")
    print(f"bundle keys           : {sorted(bundle.keys())}")
    print(f"train shape           : {tuple(bundle['train'].shape)}")
    print(f"reference shape       : {tuple(bundle['reference'].shape)}")
