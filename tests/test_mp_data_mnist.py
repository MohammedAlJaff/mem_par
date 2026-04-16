import unittest
from unittest import mock

try:
    import torch
    from mp_config import clone_config, get_base_config
    from mp_data.mnist import load_mnist_bundle
except ModuleNotFoundError:
    torch = None
    clone_config = None
    get_base_config = None
    load_mnist_bundle = None


class FakeMNIST:
    def __init__(self, root=None, train=True, download=False):
        count = 60 if train else 40
        offset = 0 if train else 100
        values = torch.arange(count * 28 * 28, dtype=torch.float32).view(count, 28, 28)
        self.data = ((values + offset) % 255).to(torch.uint8)
        self.targets = torch.arange(count) % 10


@unittest.skipIf(
    torch is None or clone_config is None or load_mnist_bundle is None,
    "torch and the MNIST data loader are required",
)
class TestMNISTLoader(unittest.TestCase):
    def make_config(self):
        config = clone_config(get_base_config())
        config.data.dataset = "mnist"
        config.data.root = "tmp/data"
        config.data.download = False
        config.data.n_train = 20
        config.data.n_eval = 20
        config.data.image_size = 32
        config.data.image_channels = 1
        config.data.flatten = False
        config.data.balance_train_classes = True
        config.data.balance_reference_classes = True
        config.training.seed = 7
        return config

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_balanced_subsets_have_exact_per_class_counts(self):
        bundle = load_mnist_bundle(self.make_config())

        self.assertEqual(bundle["train"].shape[0], 20)
        self.assertEqual(bundle["reference"].shape[0], 20)
        self.assertEqual(bundle["train_class_counts"], {str(i): 2 for i in range(10)})
        self.assertEqual(bundle["reference_class_counts"], {str(i): 2 for i in range(10)})
        self.assertEqual(bundle["train_labels"].shape[0], 20)
        self.assertEqual(bundle["reference_labels"].shape[0], 20)
        self.assertEqual(bundle["train_indices"].shape[0], 20)
        self.assertEqual(bundle["reference_indices"].shape[0], 20)

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_balanced_selection_requires_divisibility_by_ten(self):
        config = self.make_config()
        config.data.n_train = 21

        with self.assertRaisesRegex(ValueError, "divisible by 10"):
            load_mnist_bundle(config)


if __name__ == "__main__":
    unittest.main()
