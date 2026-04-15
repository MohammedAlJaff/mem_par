import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from experiments.mnist import build_mnist_config, run_mnist_study
except ModuleNotFoundError:
    torch = None
    build_mnist_config = None
    run_mnist_study = None


class FakeMNIST:
    def __init__(self, root=None, train=True, download=False):
        count = 32 if train else 24
        offset = 0 if train else 50
        values = torch.arange(count * 28 * 28, dtype=torch.float32).view(count, 28, 28)
        self.data = ((values + offset) % 255).to(torch.uint8)
        self.targets = torch.arange(count) % 10


@unittest.skipIf(
    torch is None or build_mnist_config is None or run_mnist_study is None,
    "torch and the live MNIST experiment runner are required",
)
class TestMNISTExperimentRunner(unittest.TestCase):
    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_saved_run_path_writes_artifacts_and_metrics(self):
        config = build_mnist_config(
            seed=7,
            download=False,
            n_train=4,
            n_eval=4,
            n_steps=2,
            sampling_steps=2,
            batch_size=2,
            eval_samples=2,
            run_name="mnist_runner_smoke",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir) / "mnist_study"
            run_dir, metrics = run_mnist_study(study_dir, device="cpu", config=config)

            self.assertTrue((run_dir / "artifacts" / "config.json").exists())
            self.assertTrue((run_dir / "metrics" / "train_history.json").exists())
            self.assertTrue((run_dir / "metrics" / "checkpoint_metrics.json").exists())
            self.assertTrue((study_dir / "runs" / "LATEST").exists())
            self.assertGreater(len(metrics), 0)
            self.assertIn("f_mem", metrics[-1])
            self.assertIn("baseline_f_mem", metrics[-1])


if __name__ == "__main__":
    unittest.main()
