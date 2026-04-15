import torch
from torchvision import datasets, transforms

from mp_data.common import maybe_flatten, seeded_indices


def _build_transform():
    return transforms.Compose(
        [
            transforms.CenterCrop(140),
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def _stack_subset(dataset, indices):
    images = []
    labels = []
    for index in indices.tolist():
        image, label = dataset[index]
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.as_tensor(labels)


def load_celeba32_bundle(config):
    root = config.data.root
    download = bool(config.data.download)
    transform = _build_transform()

    train_set = datasets.CelebA(
        root=root,
        split="train",
        target_type="identity",
        transform=transform,
        download=download,
    )
    reference_set = datasets.CelebA(
        root=root,
        split=config.data.reference_split,
        target_type="identity",
        transform=transform,
        download=download,
    )

    train_indices = seeded_indices(
        len(train_set), int(config.data.n_train), int(config.training.seed)
    )
    reference_indices = seeded_indices(
        len(reference_set), int(config.data.n_eval), int(config.training.seed) + 1
    )

    train_tensor, train_labels = _stack_subset(train_set, train_indices)
    reference_tensor, reference_labels = _stack_subset(reference_set, reference_indices)

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
