import json
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from prototyping.runners import (
        celeba32_jit,
        celeba32_unet,
        mnist_unet,
        synthetic_mlp,
    )
except ModuleNotFoundError:
    torch = None
    celeba32_jit = None
    celeba32_unet = None
    mnist_unet = None
    synthetic_mlp = None


ROOT = Path(__file__).resolve().parents[1]
RECIPE_RUNNERS = (
    ("mnist_unet", mnist_unet, "mnist", "unet"),
    ("celeba32_unet", celeba32_unet, "celeba32_grey", "unet"),
    ("celeba32_jit", celeba32_jit, "celeba32_grey", "jit"),
)


class FakeMNIST:
    def __init__(self, root=None, train=True, download=False):
        count = 40 if train else 30
        offset = 0 if train else 70
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
        self.length = 12 if split == "train" else 10

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = torch.linspace(-1.0, 1.0, 32 * 32).view(1, 32, 32)
        image = torch.roll(image, shifts=index % 5, dims=-1)
        return image, index % 5


@unittest.skipIf(torch is None, "torch and prototype runner dependencies are required")
class TestProtoRunners(unittest.TestCase):
    def test_image_recipe_function_order_is_visible(self):
        expected_order = [
            "def default_knobs(",
            "def build_config(",
            "def build_data(",
            "def build_model(",
            "def run_one_update(",
            "def inspect_metrics(",
            "def run_full_job(",
            "def parse_args(",
            "def main(",
        ]
        for name, module, _, _ in RECIPE_RUNNERS:
            text = Path(module.__file__).read_text()
            positions = [text.index(marker) for marker in expected_order]
            self.assertEqual(positions, sorted(positions), name)

    def test_image_default_knobs_build_expected_combo(self):
        for name, module, dataset, arch in RECIPE_RUNNERS:
            knobs = module.default_knobs()
            config = module.build_config(knobs)
            self.assertEqual(config.data.dataset, dataset, name)
            self.assertEqual(config.network.arch, arch, name)
            self.assertEqual(config.training.batch_cap, knobs["batch_cap"])
            self.assertEqual(config.training.batch_size, knobs["batch_cap"])

    def test_image_runners_accept_parameterization_spaces(self):
        for name, module, _, _ in RECIPE_RUNNERS:
            for space in ("x", "eps", "v"):
                knobs = module.default_knobs()
                knobs.update(
                    {
                        "pred_space": space,
                        "loss_space": space,
                        "n_train": 10 if name == "mnist_unet" else 4,
                        "n_eval": 10 if name == "mnist_unet" else 4,
                        "n_steps": 2,
                        "ckpt_every": 1,
                    }
                )
                config = module.build_config(knobs)
                self.assertEqual(config.framework.prediction_space, space)
                self.assertEqual(config.framework.loss_space, space)

    def test_synthetic_runner_is_direct_old_style(self):
        text = Path(synthetic_mlp.__file__).read_text()
        self.assertIn("def main(argv=None):", text)
        self.assertIn("----- training -----", text)
        self.assertIn("metrics_along_training.json", text)
        self.assertIn("wandb.init", text)
        self.assertNotIn("from mp_training import train_run", text)
        self.assertNotIn("from mp_metrics import evaluate_run", text)

    def test_synthetic_validate_only_includes_equivalence_and_oracle_metrics(self):
        result = synthetic_mlp.main(
            [
                "--validate-only",
                "--device",
                "cpu",
                "--n-train",
                "8",
                "--n-reference",
                "12",
                "--batch-size",
                "4",
                "--eval-samples",
                "4",
                "--sampling-steps",
                "2",
                "--data-dim",
                "8",
                "--intrinsic-dim",
                "2",
                "--n-clusters",
                "3",
                "--hidden-dim",
                "16",
                "--n-layers",
                "2",
                "--time-emb-dim",
                "16",
                "--sliced-w2-projections",
                "8",
                "--n-steps",
                "2",
                "--metric-every",
                "1",
                "--ckpt-every",
                "1",
            ]
        )

        self.assertEqual(result["mode"], "validate_only")
        self.assertGreater(result["one_update"]["loss"], 0.0)
        self.assertTrue(result["manual_vs_canonical"]["all_equal"])
        self.assertIn("oracle_latent_nll_mean", result["metrics"])
        self.assertIn("ambient_f_mem", result["metrics"])
        self.assertEqual(result["batching"]["batch_cap"], 4)
        self.assertEqual(result["batching"]["effective_batch_size"], 4)
        self.assertTrue(result["batching"]["drop_last"])

    def test_synthetic_accepts_parameterization_spaces(self):
        for space in ("x", "eps", "v"):
            args = synthetic_mlp.parse_args(
                [
                    "--prediction-space",
                    space,
                    "--loss-space",
                    space,
                ]
            )
            self.assertEqual(args.prediction_space, space)
            self.assertEqual(args.loss_space, space)

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_mnist_unet_recipe_steps_use_fake_dataset(self):
        knobs = mnist_unet.default_knobs()
        knobs.update(
            {
                "n_train": 10,
                "n_eval": 10,
                "n_steps": 2,
                "ckpt_every": 1,
                "batch_cap": 2,
                "eval_samples": 2,
                "sampling_steps": 2,
                "image_size": 16,
                "base_channels": 8,
                "channel_mult": (1, 2),
                "attn_resolutions": (8,),
                "num_res_blocks": 1,
                "time_emb_dim": 16,
                "download": False,
            }
        )
        config = mnist_unet.build_config(knobs)
        self.assertEqual(config.training.batch_cap, 2)
        data = mnist_unet.build_data(config)
        model = mnist_unet.build_model(config, "cpu")
        update = mnist_unet.run_one_update(model, data["train"], config, "cpu")
        metrics = mnist_unet.inspect_metrics(config, data, model, "cpu")

        self.assertEqual(list(data["train"].shape), [10, 1, 16, 16])
        self.assertEqual(model.net.__class__.__name__, "BonnaireUNet")
        self.assertGreater(update["loss"], 0.0)
        self.assertIn("f_mem", metrics["reference_memory"])

    @mock.patch("mp_data.celeba32.datasets.CelebA", FakeCelebA)
    def test_celeba_unet_recipe_steps_use_fake_dataset(self):
        knobs = celeba32_unet.default_knobs()
        knobs.update(
            {
                "n_train": 4,
                "n_eval": 4,
                "n_steps": 2,
                "ckpt_every": 1,
                "batch_cap": 2,
                "eval_samples": 2,
                "sampling_steps": 2,
                "base_channels": 8,
                "channel_mult": (1, 2),
                "attn_resolutions": (8,),
                "num_res_blocks": 1,
                "time_emb_dim": 16,
                "download": False,
            }
        )
        config = celeba32_unet.build_config(knobs)
        self.assertEqual(config.training.batch_cap, 2)
        data = celeba32_unet.build_data(config)
        model = celeba32_unet.build_model(config, "cpu")
        update = celeba32_unet.run_one_update(model, data["train"], config, "cpu")
        metrics = celeba32_unet.inspect_metrics(config, data, model, "cpu")

        self.assertEqual(list(data["train"].shape), [4, 1, 32, 32])
        self.assertEqual(model.net.__class__.__name__, "BonnaireUNet")
        self.assertGreater(update["loss"], 0.0)
        self.assertIn("f_mem", metrics["reference_memory"])

    @mock.patch("mp_data.celeba32.datasets.CelebA", FakeCelebA)
    def test_celeba_jit_recipe_steps_use_fake_dataset(self):
        knobs = celeba32_jit.default_knobs()
        knobs.update(
            {
                "n_train": 2,
                "n_eval": 2,
                "n_steps": 2,
                "ckpt_every": 1,
                "batch_cap": 1,
                "eval_samples": 1,
                "sampling_steps": 2,
                "image_size": 32,
                "patch_size": 4,
                "hidden_dim": 32,
                "depth": 2,
                "heads": 4,
                "bottleneck_dim": 16,
                "download": False,
            }
        )
        config = celeba32_jit.build_config(knobs)
        self.assertEqual(config.training.batch_cap, 1)
        data = celeba32_jit.build_data(config)
        model = celeba32_jit.build_model(config, "cpu")
        update = celeba32_jit.run_one_update(model, data["train"], config, "cpu")
        metrics = celeba32_jit.inspect_metrics(config, data, model, "cpu")

        self.assertEqual(list(data["train"].shape), [2, 1, 32, 32])
        self.assertEqual(model.net.__class__.__name__, "JiT")
        self.assertGreater(update["loss"], 0.0)
        self.assertIn("f_mem", metrics["reference_memory"])

    def test_gmm_compatibility_aliases_parse(self):
        args = synthetic_mlp.parse_args(
            [
                "--pred-space",
                "eps",
                "--loss-space",
                "eps",
                "--n-reference",
                "11",
                "--out-dir",
                "tmp/alias_check",
            ]
        )
        self.assertEqual(args.prediction_space, "eps")
        self.assertEqual(args.loss_space, "eps")
        self.assertEqual(args.n_reference, 11)
        self.assertEqual(args.out_dir, "tmp/alias_check")

    def test_batch_size_aliases_fill_batch_cap(self):
        synthetic_args = synthetic_mlp.parse_args(["--batch-size", "5"])
        self.assertEqual(synthetic_args.batch_cap, 5)

        for _, module, _, _ in RECIPE_RUNNERS:
            args = module.parse_args(["--batch-size", "3"])
            self.assertEqual(args.batch_cap, 3)
            knobs = module.default_knobs()
            knobs.update(vars(args))
            config = module.build_config(knobs)
            self.assertEqual(config.training.batch_cap, 3)
            self.assertEqual(config.training.batch_size, 3)

    def test_notebooks_have_required_sections_and_no_outputs(self):
        recipe_sections = [
            "## 0. Knobs",
            "## 1. Imports",
            "## 2. Config Construction",
            "## 3. Data Bundle",
            "## 4. Model Construction",
            "## 5. One Forward Loss Update",
            "## 6. Metric Calculation",
            "## 7. Rendered CLI Command",
            "## 8. Optional Full Job",
            "## 9. Artifact Figure Loading",
        ]
        synthetic_sections = [
            "## 0. Knobs",
            "## 1. Imports",
            "## 2. Output Directories",
            "## 3. Config",
            "## 4. Manual And Canonical Data",
            "## 5. Projection And Baselines",
            "## 6. Model And One Update",
            "## 7. Metrics",
            "## 8. Rendered CLI Command",
            "## 9. Optional Full Run",
            "## 10. Artifacts And Figures",
        ]
        notebook_checks = [
            (ROOT / "prototyping/runners/notebooks/synthetic_mlp.ipynb", synthetic_sections),
            (ROOT / "prototyping/runners/notebooks/mnist_unet.ipynb", recipe_sections),
            (ROOT / "prototyping/runners/notebooks/celeba32_unet.ipynb", recipe_sections),
            (ROOT / "prototyping/runners/notebooks/celeba32_jit.ipynb", recipe_sections),
        ]
        for notebook_path, required_sections in notebook_checks:
            notebook = json.loads(notebook_path.read_text())
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
            )
            for section in required_sections:
                self.assertIn(section, source, notebook_path.name)
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
