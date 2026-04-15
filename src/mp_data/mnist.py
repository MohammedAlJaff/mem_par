import torch
from torchvision import datasets

from mp_data.common import maybe_flatten, seeded_indices


def load_mnist_bundle(config):
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
