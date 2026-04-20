import tempfile
import unittest
from json import loads
from pathlib import Path
from unittest import mock

try:
    import torch
    from mp_config import clone_config, get_base_config
    from mp_metrics import compute_memorization_metrics, evaluate_run
    from mp_training import train_run
    from mp_training.trainer import _batching_metadata
except ModuleNotFoundError:
    torch = None
    clone_config = None
    get_base_config = None
    compute_memorization_metrics = None
    evaluate_run = None
    train_run = None
    _batching_metadata = None


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


@unittest.skipIf(
    torch is None or clone_config is None or evaluate_run is None or train_run is None,
    "torch and the first-party training stack are required",
)
class TestSrcPipeline(unittest.TestCase):
    def _run_pipeline(self, config, study_name, device="cpu"):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir) / study_name
            run_dir = train_run(config, study_dir, device=device)
            metrics = evaluate_run(run_dir, device=device)

            self.assertTrue((run_dir / "artifacts" / "config.json").exists())
            self.assertTrue((run_dir / "metrics" / "train_history.json").exists())
            self.assertTrue((run_dir / "metrics" / "checkpoint_metrics.json").exists())
            self.assertTrue((run_dir / "metrics" / "loss_eval.json").exists())
            self.assertTrue(
                (run_dir / "metrics" / "checkpoint_sample_previews.json").exists()
            )
            self.assertTrue((run_dir / "status.json").exists())
            self.assertTrue((run_dir / "logs" / "events.jsonl").exists())
            self.assertTrue((run_dir / "logs" / "attempt.log").exists())
            self.assertTrue((run_dir / "figures" / "training_loss_vs_step.png").exists())
            self.assertTrue((run_dir / "figures" / "f_mem_vs_tau.png").exists())
            self.assertTrue((run_dir / "figures" / "nn_ratio_vs_tau.png").exists())
            status = loads((run_dir / "status.json").read_text())
            data_meta = loads((run_dir / "artifacts" / "data_meta.json").read_text())
            run_summary = loads((run_dir / "artifacts" / "run_summary.json").read_text())
            train_history = loads((run_dir / "metrics" / "train_history.json").read_text())
            loss_eval = loads((run_dir / "metrics" / "loss_eval.json").read_text())
            checkpoint_previews = loads(
                (run_dir / "metrics" / "checkpoint_sample_previews.json").read_text()
            )
            self.assertEqual(status["state"], "completed")
            self.assertIn("progress", status)
            self.assertIn("evaluation_progress", status)
            self.assertIn("latest_metrics", status)
            self.assertIn("batching", status)
            self.assertIn("batching", data_meta)
            self.assertIn("batching", run_summary)
            self.assertEqual(status["batching"], data_meta["batching"])
            self.assertEqual(run_summary["batching"], data_meta["batching"])
            self.assertEqual(train_history[-1]["step"], int(config.training.n_steps))
            self.assertGreater(len(loss_eval), 0)
            self.assertIn("random_t_loss", loss_eval[-1])
            self.assertIn("loss_t_0_5", loss_eval[-1])
            self.assertIn("loss_t_0_75", loss_eval[-1])
            self.assertEqual(run_summary["final_loss_eval"], loss_eval[-1])
            self.assertGreater(len(checkpoint_previews), 0)
            self.assertEqual(
                run_summary["checkpoint_sample_previews"],
                checkpoint_previews,
            )
            self.assertTrue(Path(checkpoint_previews[-1]["samples_path"]).is_file())
            if bool(getattr(config.visual_sanity, "enabled", False)):
                figure_path = Path(checkpoint_previews[-1]["figure_path"])
                self.assertTrue(figure_path.is_file())
                figure_name = figure_path.stem
                self.assertIn(figure_name, status["generated_figures"])
                live_previews = run_summary["visual_sanity"]["live_checkpoint_samples"]
                self.assertGreater(len(live_previews), 0)
                self.assertTrue(Path(live_previews[-1]["path"]).is_file())
                self.assertIn(Path(live_previews[-1]["path"]).stem, status["generated_figures"])
            self.assertIn("training_loss_vs_step", status["generated_figures"])
            self.assertGreater(len(metrics), 0)

    def make_gmm_config(self):
        config = clone_config(get_base_config())
        config.study.name = "gmm_smoke"
        config.outputs.run_name = "gmm_smoke"
        config.data.dataset = "gmm_subspace"
        config.data.n_train = 8
        config.data.n_eval = 8
        config.data.data_dim = 16
        config.data.intrinsic_dim = 4
        config.data.n_clusters = 4
        config.network.arch = "mlp"
        config.network.hidden_dim = 32
        config.network.n_layers = 2
        config.network.time_emb_dim = 16
        config.training.n_steps = 2
        config.training.ckpt_every = 1
        config.training.log_every = 1
        config.tracking.console.enabled = False
        config.training.batch_cap = 4
        config.training.batch_size = 4
        config.evaluation.n_samples = 4
        config.evaluation.sample_panel_count = 2
        config.evaluation.chunk_size = 8
        config.sampling.n_steps = 2
        return config

    def make_mnist_config(self):
        config = clone_config(get_base_config())
        config.study.name = "mnist_smoke"
        config.outputs.run_name = "mnist_smoke"
        config.data.dataset = "mnist"
        config.data.root = "data"
        config.data.download = False
        config.data.n_train = 4
        config.data.n_eval = 4
        config.data.image_size = 28
        config.data.image_channels = 1
        config.data.flatten = True
        config.data.data_dim = 28 * 28
        config.network.arch = "mlp"
        config.training.n_steps = 2
        config.training.ckpt_every = 1
        config.training.log_every = 1
        config.tracking.console.enabled = False
        config.training.batch_cap = 4
        config.training.batch_size = 4
        config.evaluation.n_samples = 2
        config.evaluation.sample_panel_count = 2
        config.evaluation.chunk_size = 4
        config.sampling.n_steps = 2
        return config

    def make_celeba_config(self):
        config = clone_config(get_base_config())
        config.study.name = "celeba_smoke"
        config.outputs.run_name = "celeba_smoke"
        config.data.dataset = "celeba32_grey"
        config.data.root = "data"
        config.data.download = False
        config.data.reference_split = "valid"
        config.data.n_train = 4
        config.data.n_eval = 4
        config.data.image_size = 32
        config.data.image_channels = 1
        config.network.arch = "unet"
        config.network.base_channels = 8
        config.network.channel_mult = (1, 2)
        config.network.attn_resolutions = (8,)
        config.network.num_res_blocks = 1
        config.network.time_emb_dim = 32
        config.training.n_steps = 1
        config.training.ckpt_every = 1
        config.training.log_every = 1
        config.tracking.console.enabled = False
        config.training.batch_cap = 2
        config.training.batch_size = 2
        config.evaluation.n_samples = 2
        config.evaluation.sample_panel_count = 2
        config.evaluation.chunk_size = 4
        config.sampling.n_steps = 2
        config.visual_sanity.enabled = True
        config.visual_sanity.num_images = 2
        config.visual_sanity.n_updates = 0
        config.visual_sanity.show = False
        return config

    def test_memorization_metric_prefers_copies(self):
        reference = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], dtype=torch.float32)
        copied = reference[:2].clone()
        shifted = copied + 5.0

        copied_metrics = compute_memorization_metrics(copied, reference, ratio_threshold=0.5)
        shifted_metrics = compute_memorization_metrics(shifted, reference, ratio_threshold=0.5)

        self.assertGreaterEqual(copied_metrics["f_mem"], shifted_metrics["f_mem"])
        self.assertLessEqual(copied_metrics["mean_d1"], shifted_metrics["mean_d1"])

    def test_batching_metadata_small_equal_and_large_n(self):
        config = clone_config(get_base_config())
        config.training.n_steps = 7
        config.training.batch_cap = 4
        config.training.batch_size = 4

        small = _batching_metadata(config, torch.zeros(3, 2))
        self.assertEqual(small["batch_cap"], 4)
        self.assertEqual(small["effective_batch_size"], 3)
        self.assertFalse(small["drop_last"])
        self.assertEqual(small["steps_per_epoch"], 1)
        self.assertEqual(small["examples_per_epoch"], 3)
        self.assertEqual(small["dropped_per_epoch"], 0)
        self.assertEqual(small["n_steps"], 7)

        equal = _batching_metadata(config, torch.zeros(4, 2))
        self.assertEqual(equal["effective_batch_size"], 4)
        self.assertFalse(equal["drop_last"])
        self.assertEqual(equal["steps_per_epoch"], 1)
        self.assertEqual(equal["dropped_per_epoch"], 0)

        large = _batching_metadata(config, torch.zeros(10, 2))
        self.assertEqual(large["effective_batch_size"], 4)
        self.assertTrue(large["drop_last"])
        self.assertEqual(large["steps_per_epoch"], 2)
        self.assertEqual(large["examples_per_epoch"], 8)
        self.assertEqual(large["dropped_per_epoch"], 2)

    def test_gmm_runtime_smoke(self):
        self._run_pipeline(self.make_gmm_config(), "gmm_subspace")

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_mnist_runtime_smoke(self):
        self._run_pipeline(self.make_mnist_config(), "mnist_runtime")

    @mock.patch("mp_data.celeba32.datasets.CelebA", FakeCelebA)
    def test_celeba_runtime_smoke(self):
        self._run_pipeline(self.make_celeba_config(), "celeba_runtime")


if __name__ == "__main__":
    unittest.main()
