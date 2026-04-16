import tempfile
import unittest
from json import loads
from pathlib import Path
from unittest import mock

try:
    import torch
    from experiments.mnist import build_mnist_config, run_mnist_study
    from mp_artifacts import find_latest_attempt_dir, find_latest_completed_attempt_dir, make_run_id
except ModuleNotFoundError:
    torch = None
    build_mnist_config = None
    run_mnist_study = None
    find_latest_attempt_dir = None
    find_latest_completed_attempt_dir = None
    make_run_id = None


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
            n_train=20,
            n_eval=20,
            n_steps=2,
            ckpt_every=1,
            sampling_steps=2,
            batch_size=2,
            eval_samples=2,
            run_name="mnist_runner_smoke",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir) / "mnist_study"
            run_dir, metrics = run_mnist_study(study_dir, device="cpu", config=config)
            run_id = make_run_id(config)

            self.assertTrue((run_dir / "artifacts" / "config.json").exists())
            self.assertTrue((run_dir / "artifacts" / "train_labels.pt").exists())
            self.assertTrue((run_dir / "artifacts" / "reference_labels.pt").exists())
            self.assertTrue((run_dir / "artifacts" / "train_indices.pt").exists())
            self.assertTrue((run_dir / "artifacts" / "reference_indices.pt").exists())
            self.assertTrue((run_dir / "metrics" / "train_history.json").exists())
            self.assertTrue((run_dir / "metrics" / "checkpoint_metrics.json").exists())
            self.assertTrue((run_dir / "status.json").exists())
            self.assertTrue((run_dir / "figures").exists())
            self.assertTrue((run_dir / "logs" / "events.jsonl").exists())
            self.assertGreater(len(metrics), 0)
            self.assertIn("f_mem", metrics[-1])
            self.assertIn("std_nn_ratio", metrics[-1])
            self.assertIn("baseline_f_mem", metrics[-1])
            self.assertIn("baseline_std_nn_ratio", metrics[-1])

            experiment_manifest = loads((study_dir / "experiment_manifest.json").read_text())
            run_manifest = loads((run_dir.parents[3] / "run_manifest.json").read_text())
            seed_manifest = loads((run_dir.parents[1] / "seed_manifest.json").read_text())
            status = loads((run_dir / "status.json").read_text())
            data_meta = loads((run_dir / "artifacts" / "data_meta.json").read_text())
            run_summary = loads((run_dir / "artifacts" / "run_summary.json").read_text())

            self.assertEqual(run_dir.parents[3].name, run_id)
            self.assertEqual(run_dir.parents[1].name, "seed_0007")
            self.assertEqual(status["state"], "completed")
            self.assertEqual(experiment_manifest["runs"][run_id]["path"], f"runs/{run_id}")
            self.assertEqual(seed_manifest["latest_attempt"], run_dir.name)
            self.assertEqual(seed_manifest["latest_completed_attempt"], run_dir.name)
            self.assertEqual(run_manifest["seeds"]["seed_0007"]["latest_completed_attempt"], run_dir.name)
            self.assertEqual(data_meta["train_class_counts"], {str(i): 2 for i in range(10)})
            self.assertEqual(data_meta["reference_class_counts"], {str(i): 2 for i in range(10)})
            self.assertTrue(data_meta["balanced_train_classes"])
            self.assertTrue(data_meta["balanced_reference_classes"])
            self.assertIn("training_wall_time_sec", run_summary)
            self.assertIn("training_seconds_per_step", run_summary)
            self.assertGreaterEqual(run_summary["training_wall_time_sec"], 0.0)
            self.assertGreaterEqual(run_summary["training_seconds_per_step"], 0.0)
            self.assertEqual(find_latest_attempt_dir(study_dir, run_id, 7), run_dir)
            self.assertEqual(find_latest_completed_attempt_dir(study_dir, run_id, 7), run_dir)


if __name__ == "__main__":
    unittest.main()
