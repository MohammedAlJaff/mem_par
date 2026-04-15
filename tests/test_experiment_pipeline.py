import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

from experiments._shared.plot_helpers import plot_run
from experiments.celeba32_grey.config import get_config as get_celeba_config
from experiments.gmm_subspace.config import get_config as get_gmm_config
from experiments.mnist_n4_diagonal.config import get_config as get_mnist_config
from mp_config import clone_config
from mp_metrics import compute_memorization_metrics, evaluate_run
from mp_training import train_run


class FakeMNIST:
    def __init__(self, root=None, train=True, download=False):
        count = 32 if train else 24
        offset = 0 if train else 50
        values = torch.arange(count * 28 * 28, dtype=torch.float32).view(count, 28, 28)
        self.data = ((values + offset) % 255).to(torch.uint8)
        self.targets = torch.arange(count) % 10


class FakeCelebA:
    def __init__(
        self,
        root=None,
        split="train",
        target_type="identity",
        transform=None,
        download=False,
    ):
        self.split = split
        self.transform = transform
        self.length = 24 if split == "train" else 20

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = torch.linspace(-1.0, 1.0, 32 * 32).view(1, 32, 32)
        image = torch.roll(image, shifts=index % 7, dims=-1)
        label = index % 5
        return image, label


@unittest.skipIf(torch is None, "torch is required")
class TestExperimentPipeline(unittest.TestCase):
    def _run_pipeline(self, config, study_name, device="cpu"):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir) / study_name
            run_dir = train_run(config, study_dir, device=device)
            metrics = evaluate_run(run_dir, device=device)
            figures_dir = plot_run(run_dir, study_dir=study_dir)

            self.assertTrue((run_dir / "artifacts" / "config.json").exists())
            self.assertTrue((run_dir / "metrics" / "train_history.json").exists())
            self.assertTrue((run_dir / "metrics" / "checkpoint_metrics.json").exists())
            self.assertTrue((figures_dir / f"{run_dir.name}_metrics.png").exists())
            self.assertGreater(len(metrics), 0)

    def test_memorization_metric_prefers_copies(self):
        reference = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], dtype=torch.float32)
        copied = reference[:2].clone()
        shifted = copied + 5.0

        copied_metrics = compute_memorization_metrics(copied, reference, ratio_threshold=0.5)
        shifted_metrics = compute_memorization_metrics(shifted, reference, ratio_threshold=0.5)

        self.assertGreaterEqual(copied_metrics["f_mem"], shifted_metrics["f_mem"])
        self.assertLessEqual(copied_metrics["mean_d1"], shifted_metrics["mean_d1"])

    def test_gmm_study_smoke(self):
        config = clone_config(get_gmm_config())
        config.outputs.run_name = "gmm_smoke"
        config.training.n_steps = 2
        config.training.ckpt_every = 1
        config.training.log_every = 1
        config.training.batch_size = 4
        config.data.n_train = 8
        config.data.n_eval = 8
        config.evaluation.n_samples = 4
        config.evaluation.sample_panel_count = 2
        config.evaluation.chunk_size = 8
        config.sampling.n_steps = 2
        self._run_pipeline(config, "gmm_subspace")

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_mnist_study_smoke(self):
        config = clone_config(get_mnist_config())
        config.outputs.run_name = "mnist_smoke"
        config.training.n_steps = 2
        config.training.ckpt_every = 1
        config.training.log_every = 1
        config.training.batch_size = 4
        config.data.n_train = 4
        config.data.n_eval = 4
        config.evaluation.n_samples = 2
        config.evaluation.sample_panel_count = 2
        config.evaluation.chunk_size = 4
        config.sampling.n_steps = 2
        self._run_pipeline(config, "mnist_n4_diagonal")

    @mock.patch("mp_data.celeba32.datasets.CelebA", FakeCelebA)
    def test_celeba_study_smoke(self):
        config = clone_config(get_celeba_config())
        config.outputs.run_name = "celeba_smoke"
        config.network.base_channels = 8
        config.network.channel_mult = (1, 2)
        config.network.attn_resolutions = (8,)
        config.network.num_res_blocks = 1
        config.network.time_emb_dim = 32
        config.training.n_steps = 1
        config.training.ckpt_every = 1
        config.training.log_every = 1
        config.training.batch_size = 2
        config.data.n_train = 4
        config.data.n_eval = 4
        config.evaluation.n_samples = 2
        config.evaluation.sample_panel_count = 2
        config.evaluation.chunk_size = 4
        config.sampling.n_steps = 2
        self._run_pipeline(config, "celeba32_grey")


if __name__ == "__main__":
    unittest.main()
