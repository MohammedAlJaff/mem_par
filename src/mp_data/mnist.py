"""
Canonical MNIST subset loader for first-party studies.

This loader intentionally keeps the data path simple:
- sample deterministic subsets from the torchvision train and test splits
- normalize pixels to `[-1, 1]`
- optionally flatten the images when a vector-valued backbone is requested
"""

import torch
from torchvision import datasets

from mp_data.common import maybe_flatten, seeded_indices


def load_mnist_bundle(config):
    """
    Return a deterministic MNIST train/reference bundle plus labels and indices.

    The training subset comes from the torchvision training split. The
    reference subset comes from the test split so memorization sanity checks can
    compare generated samples against held-out real data as well.

    This is a good example of the shared data-bundle contract in practice:
    MNIST may be simple, but it still returns the same train/reference/labels/
    indices structure as the higher-dimensional studies.
    """
    root = config.data.root
    download = bool(config.data.download)

    train_set = datasets.MNIST(root=root, train=True, download=download)
    test_set = datasets.MNIST(root=root, train=False, download=download)

    train_indices = seeded_indices(
        len(train_set.data), int(config.data.n_train), int(config.training.seed)
    )
    reference_indices = seeded_indices(
        len(test_set.data), int(config.data.n_eval), int(config.training.seed) + 1
    )

    train_tensor = train_set.data[train_indices].float().div(127.5).sub(1.0).unsqueeze(1)
    reference_tensor = (
        test_set.data[reference_indices].float().div(127.5).sub(1.0).unsqueeze(1)
    )
    train_labels = train_set.targets[train_indices].clone()
    reference_labels = test_set.targets[reference_indices].clone()

    train_tensor = maybe_flatten(train_tensor, config)
    reference_tensor = maybe_flatten(reference_tensor, config)

    return {
        "train": train_tensor.contiguous(),
        "train_labels": train_labels,
        "reference": reference_tensor.contiguous(),
        "reference_labels": reference_labels,
        "train_indices": train_indices,
        "reference_indices": reference_indices,
        "display_shape": (
            int(config.data.image_channels),
            int(config.data.image_size),
            int(config.data.image_size),
        ),
    }


if __name__ == "__main__":
    from pathlib import Path

    from mp_config import get_base_config

    print("mp_data.mnist example")

    # Build the smallest MNIST config that still shows the loader contract.
    config = get_base_config()
    config.data.dataset = "mnist"
    config.data.root = "tmp/data"
    config.data.n_train = 8
    config.data.n_eval = 6
    config.data.image_size = 28
    config.data.image_channels = 1
    config.data.flatten = False
    config.data.download = False

    # Show the direct call pattern before attempting any local filesystem use.
    print("Usage:")
    print("  config = get_base_config()")
    print("  config.data.dataset = 'mnist'")
    print("  bundle = load_mnist_bundle(config)")

    # Skip cleanly when local data is absent. This example should stay safe to
    # run on a fresh machine without triggering a download.
    if not Path(config.data.root).exists():
        print(f"Skipping runnable demo: {config.data.root!r} does not exist.")
    else:
        try:
            # Load one tiny bundle and print the tensor shapes we expect.
            bundle = load_mnist_bundle(config)
            print(f"train shape     : {tuple(bundle['train'].shape)}")
            print(f"reference shape : {tuple(bundle['reference'].shape)}")
        except Exception as exc:
            print(f"Skipping runnable demo: {exc}")
