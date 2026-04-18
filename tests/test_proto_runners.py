import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from mp_data import build_data_bundle
    from prototyping.runners import (
        image32_unet,
        synthetic_mlp,
    )
except ModuleNotFoundError:
    torch = None
    build_data_bundle = None
    image32_unet = None
    synthetic_mlp = None


ROOT = Path(__file__).resolve().parents[1]
RECIPE_RUNNERS = (
    ("image32_unet", image32_unet, "celeba32_grey", "unet"),
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


def manual_projected_gmm(args):
    geometry_generator = torch.Generator().manual_seed(int(args.geometry_seed))
    basis, _ = torch.linalg.qr(
        torch.randn(args.data_dim, args.intrinsic_dim, generator=geometry_generator)
    )
    basis = basis[:, : args.intrinsic_dim]
    means = torch.randn(args.n_clusters, args.intrinsic_dim, generator=geometry_generator)

    train_generator = torch.Generator().manual_seed(int(args.seed))
    train_labels = torch.randint(args.n_clusters, (args.n_train,), generator=train_generator)
    train_latent = means[train_labels] + args.sigma_z * torch.randn(
        args.n_train,
        args.intrinsic_dim,
        generator=train_generator,
    )
    train = train_latent @ basis.T
    train = train + args.sigma_eta * torch.randn(
        args.n_train,
        args.data_dim,
        generator=train_generator,
    )

    reference_generator = torch.Generator().manual_seed(int(args.seed) + 1)
    reference_labels = torch.randint(
        args.n_clusters,
        (args.n_reference,),
        generator=reference_generator,
    )
    reference_latent = means[reference_labels] + args.sigma_z * torch.randn(
        args.n_reference,
        args.intrinsic_dim,
        generator=reference_generator,
    )
    reference = reference_latent @ basis.T
    reference = reference + args.sigma_eta * torch.randn(
        args.n_reference,
        args.data_dim,
        generator=reference_generator,
    )
    return {
        "train": train.contiguous(),
        "reference": reference.contiguous(),
        "train_labels": train_labels,
        "reference_labels": reference_labels,
        "basis": basis.contiguous(),
        "means": means.contiguous(),
    }


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
                        "prediction_space": space,
                        "loss_space": space,
                        "n_train": 4,
                        "n_eval": 4,
                        "n_steps": 2,
                        "ckpt_every": 1,
                    }
                )
                config = module.build_config(knobs)
                self.assertEqual(config.framework.prediction_space, space)
                self.assertEqual(config.framework.loss_space, space)

    def test_release_clean_flags_are_removed(self):
        removed = [
            (synthetic_mlp, ["--batch-size", "5"]),
            (synthetic_mlp, ["--validate-only"]),
            (synthetic_mlp, ["--data-source", "both"]),
            (synthetic_mlp, ["--pred-space", "eps"]),
        ]
        for _, module, _, _ in RECIPE_RUNNERS:
            removed.extend(
                [
                    (module, ["--batch-size", "3"]),
                    (module, ["--validate-only"]),
                    (module, ["--pred-space", "eps"]),
                    (module, ["--study-dir", "tmp/old"]),
                ]
            )
        for module, argv in removed:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.parse_args(argv)

    def test_synthetic_accepts_release_clean_public_flags(self):
        args = synthetic_mlp.parse_args(
            [
                "--prediction-space",
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

    def test_synthetic_runner_keeps_infrastructure_out_of_main_file(self):
        text = Path(synthetic_mlp.__file__).read_text()
        self.assertIn("def main(argv=None):", text)
        self.assertIn("collect_metric_record", text)
        self.assertIn("metrics_along_training.json", text)
        self.assertNotIn("json.dumps", text)
        self.assertNotIn("wandb.init", text)
        self.assertNotIn("plt.subplots", text)
        self.assertNotIn("from mp_training import train_run", text)
        self.assertNotIn("from mp_metrics import evaluate_run", text)

    def test_image32_runner_is_the_only_image_recipe_surface(self):
        text = Path(image32_unet.__file__).read_text()
        self.assertIn("metrics_image32", text)
        self.assertNotIn("mnist_unet", text)
        self.assertNotIn("celeba32_unet", text)
        self.assertNotIn("celeba32_jit", text)

    def test_manual_gmm_equations_match_canonical_builder(self):
        args = synthetic_mlp.parse_args(
            [
                "--device",
                "cpu",
                "--n-train",
                "8",
                "--n-reference",
                "12",
                "--batch-cap",
                "4",
                "--data-dim",
                "8",
                "--intrinsic-dim",
                "2",
                "--n-clusters",
                "3",
            ]
        )
        config, _ = synthetic_mlp.build_config(args, batch_size=4)
        canonical = build_data_bundle(config)
        manual = manual_projected_gmm(args)
        self.assertTrue(torch.equal(manual["train"], canonical["train"]))
        self.assertTrue(torch.equal(manual["reference"], canonical["reference"]))
        self.assertTrue(torch.equal(manual["train_labels"], canonical["train_labels"]))
        self.assertTrue(torch.equal(manual["reference_labels"], canonical["reference_labels"]))
        self.assertTrue(torch.equal(manual["basis"], canonical["geometry"]["basis"]))
        self.assertTrue(torch.equal(manual["means"], canonical["geometry"]["means"]))

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_image32_unet_recipe_steps_use_fake_mnist_dataset(self):
        knobs = image32_unet.default_knobs()
        knobs.update(
            {
                "dataset": "mnist",
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
        config = image32_unet.build_config(knobs)
        data = image32_unet.build_data(config)
        model = image32_unet.build_model(config, "cpu")
        update = image32_unet.run_one_update(model, data["train"], config, "cpu")
        metrics = image32_unet.inspect_metrics(config, data, model, "cpu")

        self.assertEqual(list(data["train"].shape), [10, 1, 16, 16])
        self.assertEqual(model.net.__class__.__name__, "BonnaireUNet")
        self.assertGreater(update["loss"], 0.0)
        self.assertIn("f_mem", metrics["reference_memory"])
        self.assertIn("random_t_loss", metrics["loss_eval"])
        self.assertIn("loss_t_0_1", metrics["loss_eval"])

    @mock.patch("mp_data.celeba32.datasets.CelebA", FakeCelebA)
    def test_image32_unet_recipe_steps_use_fake_celeba_dataset(self):
        knobs = image32_unet.default_knobs()
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
        config = image32_unet.build_config(knobs)
        data = image32_unet.build_data(config)
        model = image32_unet.build_model(config, "cpu")
        update = image32_unet.run_one_update(model, data["train"], config, "cpu")
        metrics = image32_unet.inspect_metrics(config, data, model, "cpu")

        self.assertEqual(list(data["train"].shape), [4, 1, 32, 32])
        self.assertEqual(model.net.__class__.__name__, "BonnaireUNet")
        self.assertGreater(update["loss"], 0.0)
        self.assertIn("f_mem", metrics["reference_memory"])

    def test_synthetic_tiny_full_run_writes_release_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = synthetic_mlp.main(
                    [
                        "--device",
                        "cpu",
                        "--out-dir",
                        tmpdir,
                        "--run-name",
                        "tiny_synth",
                        "--n-train",
                        "8",
                        "--n-reference",
                        "12",
                        "--batch-cap",
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
                        "--log-every",
                        "1",
                    ]
                )
            self.assertEqual(result["batch"]["batch_cap"], 4)
            self.assertEqual(result["batch"]["effective_batch_size"], 4)
            self.assertTrue(result["batch"]["drop_last"])
            self.assertIn("ambient_f_mem", result["final_metrics"])
            self.assertIn("latent_f_mem", result["final_metrics"])
            self.assertNotIn("f_mem", result["final_metrics"])
            self.assertTrue(Path(result["paths"]["metrics_along_training"]).is_file())
            self.assertTrue(Path(result["paths"]["final_samples"]).is_file())

    def test_notebooks_have_required_sections_and_no_outputs(self):
        recipe_sections = [
            "## 0. Knobs",
            "## 1. Imports",
            "## 2. Config Construction",
            "## 3. Data Bundle",
            "## 4. Model Construction",
            "## 5. One Forward Loss Update",
            "## 6. Metric Calculation",
            "## 7. Optional Full Job",
            "## 8. Artifact Figure Loading",
        ]
        synthetic_sections = [
            "## 0. Knobs",
            "## 1. Imports",
            "## 2. Config",
            "## 3. Data",
            "## 4. Projection And Baselines",
            "## 5. Model And One Update",
            "## 6. Metrics",
            "## 7. Optional Full Run",
            "## 8. Artifacts And Figures",
        ]
        notebook_checks = [
            (ROOT / "prototyping/runners/notebooks/synthetic_mlp.ipynb", synthetic_sections),
            (ROOT / "prototyping/runners/notebooks/image32_unet.ipynb", recipe_sections),
        ]
        for notebook_path, required_sections in notebook_checks:
            notebook = json.loads(notebook_path.read_text())
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
            )
            for section in required_sections:
                self.assertIn(section, source, notebook_path.name)
            self.assertNotIn("Rendered CLI Command", source)
            self.assertNotIn("validate_only", source)
            self.assertNotIn("command_from_knobs", source)
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
