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
        image_runner,
        synthetic_mlp,
    )
except ModuleNotFoundError:
    torch = None
    build_data_bundle = None
    image_runner = None
    synthetic_mlp = None


ROOT = Path(__file__).resolve().parents[1]
RECIPE_RUNNERS = (
    ("image_runner", image_runner, "celeba32_grey", "unet"),
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
            "def default_settings(",
            "def build_config(",
            "def build_data(",
            "def build_model(",
            "def inspect_metrics(",
            "def run_full_job(",
            "def parse_args(",
            "def main(",
        ]
        for name, module, _, _ in RECIPE_RUNNERS:
            text = Path(module.__file__).read_text()
            positions = [text.index(marker) for marker in expected_order]
            self.assertEqual(positions, sorted(positions), name)

    def test_image_default_settings_build_expected_combo(self):
        for name, module, dataset, arch in RECIPE_RUNNERS:
            settings = module.default_settings()
            config = module.build_config(settings)
            self.assertEqual(settings["out_dir"], "outputs/prototype_runners/image_runner")
            self.assertEqual(config.study.name, "image_runner")
            self.assertEqual(config.data.dataset, dataset, name)
            self.assertEqual(config.network.arch, arch, name)
            self.assertEqual(config.training.batch_cap, settings["batch_cap"])
            self.assertEqual(config.training.batch_size, settings["batch_cap"])
            self.assertEqual(config.network.base_channels, 32)
            self.assertEqual(tuple(config.network.channel_mult), (1, 2, 4))
            self.assertEqual(tuple(config.network.attn_resolutions), (16, 8))
            self.assertEqual(config.network.num_res_blocks, 2)
            self.assertEqual(config.network.dropout, 0.1)
            self.assertIsNone(config.network.time_emb_dim)
            self.assertEqual(config.training.optimizer, "adam")
            self.assertEqual(config.training.lr, 1.0e-4)
            self.assertEqual(config.runtime.precision, "fp32")
            self.assertEqual(config.study.dataset_key, "celeba32_grey")
            self.assertEqual(config.tracking.wandb.project, "memparam-celeba32_grey-unet")
            self.assertFalse(settings["final_eval"])
            self.assertIsNone(settings["eval_run_dir"])

    def test_image_runner_exposes_train_loss_log_cadence(self):
        settings = image_runner.default_settings()
        settings["log_every"] = 7
        config = image_runner.build_config(settings)
        args = image_runner.parse_args(["--log-every", "7"])
        default_args = image_runner.parse_args([])

        self.assertEqual(config.training.log_every, 7)
        self.assertEqual(args.log_every, 7)
        self.assertEqual(default_args.precision, "fp32")
        self.assertFalse(default_args.compare_numerics)
        self.assertFalse(default_args.final_eval)
        self.assertIsNone(default_args.eval_run_dir)

    def test_image_runner_accepts_slim_cli_surface(self):
        args = image_runner.parse_args(
            [
                "--device",
                "cpu",
                "--out-dir",
                "tmp/image32",
                "--run-name",
                "slim",
                "--seed",
                "3",
                "--dataset",
                "mnist",
                "--config-preset",
                "image32_32",
                "--prediction-space",
                "eps",
                "--loss-space",
                "eps",
                "--n-train",
                "12",
                "--n-eval",
                "10",
                "--n-steps",
                "5",
                "--ckpt-every",
                "2",
                "--log-every",
                "1",
                "--batch-cap",
                "4",
                "--precision",
                "bf16",
                "--compare-numerics",
                "--final-eval",
                "--eval-run-dir",
                "DONE_some_attempt",
                "--eval-samples",
                "6",
                "--sampling-steps",
                "3",
                "--root",
                "data",
                "--no-download",
                "--no-wandb",
                "--wandb-mode",
                "offline",
                "--wandb-project",
                "proj",
                "--wandb-name",
                "name",
                "--wandb-group",
                "group",
                "--wandb-tags",
                "a,b",
                "--generation-trace",
                "--generation-trace-count",
                "2",
                "--no-generation-trace-show",
            ]
        )
        config = image_runner.build_config(vars(args))

        self.assertEqual(args.prediction_space, "eps")
        self.assertEqual(args.loss_space, "eps")
        self.assertFalse(args.download)
        self.assertFalse(args.wandb)
        self.assertEqual(args.precision, "bf16")
        self.assertTrue(args.compare_numerics)
        self.assertTrue(args.final_eval)
        self.assertEqual(args.eval_run_dir, "DONE_some_attempt")
        self.assertTrue(args.generation_trace)
        self.assertEqual(config.runtime.precision, "bf16")
        self.assertEqual(config.tracking.wandb.project, "proj")
        self.assertEqual(config.generation_trace.num_images, 2)
        self.assertFalse(config.generation_trace.show)
        self.assertEqual(config.tracking.wandb.tags, ["a", "b"])

    def test_image32_config_preset_owns_rare_geometry_and_architecture(self):
        settings = image_runner.default_settings()
        settings.update({"dataset": "flowers102", "config_preset": "image32_128"})

        config = image_runner.build_config(settings)

        self.assertEqual(config.data.dataset, "flowers102")
        self.assertEqual(config.study.dataset_key, "flowers102")
        self.assertEqual(config.data.train_split, "test")
        self.assertEqual(config.data.reference_split, "train")
        self.assertEqual(config.data.image_size, 128)
        self.assertEqual(config.data.resize_size, 128)
        self.assertEqual(config.data.image_channels, 3)
        self.assertEqual(config.network.base_channels, 64)
        self.assertEqual(tuple(config.network.channel_mult), (1, 2, 4, 4, 8))
        self.assertEqual(tuple(config.network.attn_resolutions), (16,))
        self.assertEqual(config.network.dropout, 0.0)

    def test_image32_wandb_project_uses_dataset_key_and_model(self):
        settings = image_runner.default_settings()
        settings.update({"dataset": "flowers102_128"})

        config = image_runner.build_config(settings)

        self.assertEqual(config.data.dataset, "flowers102")
        self.assertEqual(config.study.dataset_key, "flowers102_128")
        self.assertEqual(config.network.arch, "unet")
        self.assertEqual(config.tracking.wandb.project, "memparam-flowers102_128-unet")

    def test_image32_legacy_broad_wandb_project_routes_to_dataset_project(self):
        settings = image_runner.default_settings()
        settings.update(
            {
                "dataset": "mnist",
                "wandb_project": "memparam-image32-unet",
            }
        )

        config = image_runner.build_config(settings)

        self.assertEqual(config.tracking.wandb.project, "memparam-mnist-unet")

    def test_image_runners_accept_parameterization_spaces(self):
        for name, module, _, _ in RECIPE_RUNNERS:
            for space in ("x", "eps", "v"):
                settings = module.default_settings()
                settings.update(
                    {
                        "prediction_space": space,
                        "loss_space": space,
                        "n_train": 4,
                        "n_eval": 4,
                        "n_steps": 2,
                        "ckpt_every": 1,
                    }
                )
                config = module.build_config(settings)
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
                    (module, ["--sanity-only"]),
                    (module, ["--sanity-updates", "1"]),
                    (module, ["--base-channels", "16"]),
                    (module, ["--channel-mult", "1,2"]),
                    (module, ["--attn-resolutions", "8"]),
                    (module, ["--num-res-blocks", "1"]),
                    (module, ["--dropout", "0.0"]),
                    (module, ["--time-emb-dim", "16"]),
                    (module, ["--dataset-subdir", "tmp"]),
                    (module, ["--train-split", "train"]),
                    (module, ["--reference-split", "valid"]),
                    (module, ["--allow-same-source-reference"]),
                    (module, ["--image-size", "16"]),
                    (module, ["--image-channels", "1"]),
                    (module, ["--resize-size", "16"]),
                    (module, ["--balance-train-classes"]),
                    (module, ["--balance-reference-classes"]),
                    (module, ["--loss-eval-times", "0.5"]),
                    (module, ["--wandb-entity", "entity"]),
                    (module, ["--wandb-notes", "notes"]),
                    (module, ["--wandb-strict"]),
                    (module, ["--visual-sanity"]),
                    (module, ["--visual-sanity-count", "2"]),
                    (module, ["--visual-sanity-updates", "1"]),
                    (module, ["--visual-sanity-show"]),
                    (module, ["--generation-trace-updates", "1"]),
                    (module, ["--precision", "fp16"]),
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

    def test_image_runner_is_the_only_image_recipe_surface(self):
        text = Path(image_runner.__file__).read_text()
        self.assertIn("metrics_image32", text)
        self.assertNotIn("mnist_unet", text)
        self.assertNotIn("celeba32_unet", text)
        self.assertNotIn("celeba32_jit", text)
        self.assertFalse((ROOT / "prototyping/runners/image32_unet.py").exists())
        self.assertFalse(
            (ROOT / "prototyping/runners/image32_unet_config_builder.py").exists()
        )

    def test_image_runner_docs_use_new_public_entrypoint(self):
        docs = [
            ROOT / "README.md",
            ROOT / "prototyping/runners/README.md",
            ROOT / "prototyping/runners/notebooks/image_runner.ipynb",
        ]
        for path in docs:
            text = path.read_text()
            self.assertNotIn("prototyping.runners.image32_unet", text)
            self.assertNotIn("notebooks/image32_unet.ipynb", text)
            self.assertNotIn("outputs/prototype_runners/image32_unet", text)

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
    def test_image_runner_recipe_steps_use_fake_mnist_dataset(self):
        settings = image_runner.default_settings()
        settings.update(
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
        config = image_runner.build_config(settings)
        data = image_runner.build_data(config)
        model = image_runner.build_model(config, "cpu")
        metrics = image_runner.inspect_metrics(config, data, model, "cpu")

        self.assertEqual(list(data["train"].shape), [10, 1, 16, 16])
        self.assertEqual(model.net.__class__.__name__, "BonnaireUNet")
        self.assertIn("f_mem", metrics["reference_memory"])
        self.assertIn("random_t_loss", metrics["loss_eval"])
        self.assertIn("loss_t_0_5", metrics["loss_eval"])
        self.assertIn("loss_t_0_75", metrics["loss_eval"])

    @mock.patch("mp_data.celeba32.datasets.CelebA", FakeCelebA)
    def test_image_runner_recipe_steps_use_fake_celeba_dataset(self):
        settings = image_runner.default_settings()
        settings.update(
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
        config = image_runner.build_config(settings)
        data = image_runner.build_data(config)
        model = image_runner.build_model(config, "cpu")
        metrics = image_runner.inspect_metrics(config, data, model, "cpu")

        self.assertEqual(list(data["train"].shape), [4, 1, 32, 32])
        self.assertEqual(model.net.__class__.__name__, "BonnaireUNet")
        self.assertIn("f_mem", metrics["reference_memory"])

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_image_runner_default_run_skips_final_eval_but_keeps_generation_trace(self):
        settings = image_runner.default_settings()
        settings.update(
            {
                "dataset": "mnist",
                "n_train": 4,
                "n_eval": 4,
                "n_steps": 1,
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
                "balance_train_classes": False,
                "balance_reference_classes": False,
                "wandb": False,
                "generation_trace": True,
                "generation_trace_count": 2,
                "generation_trace_show": False,
            }
        )
        config = image_runner.build_config(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = image_runner.run_full_job(config, tmpdir, "cpu")

            self.assertEqual(result["checkpoint_metrics"], [])
            self.assertEqual(result["loss_eval"], [])
            self.assertEqual(result["checkpoint_sample_previews"], [])
            self.assertFalse(Path(result["artifacts"]["checkpoint_metrics"]["path"]).exists())
            self.assertFalse(Path(result["artifacts"]["loss_eval"]["path"]).exists())
            self.assertFalse(
                Path(result["artifacts"]["checkpoint_sample_previews"]["path"]).exists()
            )
            summary_path = Path(result["run_dir"]) / "artifacts" / "run_summary.json"
            run_summary = json.loads(summary_path.read_text())
            trace = run_summary["generation_trace"]
            self.assertTrue(Path(trace["z_init_path"]).is_file())
            self.assertTrue(Path(trace["train_grid_path"]).is_file())
            self.assertTrue(Path(trace["final_samples_path"]).is_file())
            self.assertGreater(len(trace["checkpoint_samples"]), 0)

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_image32_generation_trace_full_run_writes_checkpoint_previews(self):
        settings = image_runner.default_settings()
        settings.update(
            {
                "dataset": "mnist",
                "n_train": 4,
                "n_eval": 4,
                "n_steps": 1,
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
                "balance_train_classes": False,
                "balance_reference_classes": False,
                "wandb": False,
                "generation_trace": True,
                "generation_trace_count": 2,
                "generation_trace_show": False,
            }
        )
        config = image_runner.build_config(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = image_runner.run_full_job(config, tmpdir, "cpu", evaluate=True)

            previews = result["checkpoint_sample_previews"]
            self.assertGreater(len(previews), 0)
            self.assertTrue(Path(result["artifacts"]["checkpoint_sample_previews"]["path"]).is_file())
            self.assertTrue(Path(previews[-1]["samples_path"]).is_file())
            self.assertTrue(Path(previews[-1]["figure_path"]).is_file())
            summary_path = Path(result["run_dir"]) / "artifacts" / "run_summary.json"
            run_summary = json.loads(summary_path.read_text())
            trace = run_summary["generation_trace"]
            self.assertTrue(Path(trace["z_init_path"]).is_file())
            self.assertTrue(Path(trace["train_grid_path"]).is_file())
            self.assertTrue(Path(trace["final_samples_path"]).is_file())
            checkpoint_trace = trace["checkpoint_samples"]
            self.assertGreater(len(checkpoint_trace), 0)
            self.assertEqual(checkpoint_trace[0]["step"], 0)
            self.assertTrue(Path(checkpoint_trace[-1]["path"]).is_file())

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_image_runner_eval_run_dir_replays_finished_done_run(self):
        settings = image_runner.default_settings()
        settings.update(
            {
                "dataset": "mnist",
                "n_train": 4,
                "n_eval": 4,
                "n_steps": 1,
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
                "balance_train_classes": False,
                "balance_reference_classes": False,
                "wandb": False,
                "generation_trace": False,
            }
        )
        config = image_runner.build_config(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                train_result = image_runner.run_full_job(config, tmpdir, "cpu")
            run_dir = Path(train_result["run_dir"])
            self.assertTrue(run_dir.name.startswith("DONE_"))
            self.assertFalse(Path(train_result["artifacts"]["checkpoint_metrics"]["path"]).exists())

            with contextlib.redirect_stdout(io.StringIO()):
                eval_result = image_runner.main(
                    ["--device", "cpu", "--eval-run-dir", str(run_dir)]
                )

            self.assertEqual(eval_result["mode"], "eval_run_dir")
            self.assertEqual(Path(eval_result["run_dir"]), run_dir)
            self.assertGreater(len(eval_result["checkpoint_metrics"]), 0)
            self.assertGreater(len(eval_result["loss_eval"]), 0)
            self.assertGreater(len(eval_result["checkpoint_sample_previews"]), 0)
            self.assertTrue(Path(eval_result["artifacts"]["checkpoint_metrics"]["path"]).is_file())
            status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(status["state"], "completed")

    def test_image_runner_eval_run_dir_rejects_unfinished_or_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with self.assertRaisesRegex(ValueError, "DONE_"):
                image_runner.run_eval_job(root / "attempt_0001", "cpu")

            for state in ("failed", "aborted"):
                done_bad = root / f"DONE_{state}"
                done_bad.mkdir()
                (done_bad / "status.json").write_text(json.dumps({"state": state}))
                with self.assertRaisesRegex(ValueError, "trained"):
                    image_runner.run_eval_job(done_bad, "cpu")

            done_incomplete = root / "DONE_incomplete"
            done_incomplete.mkdir()
            (done_incomplete / "status.json").write_text(json.dumps({"state": "trained"}))
            with self.assertRaisesRegex(FileNotFoundError, "required saved-run artifacts"):
                image_runner.run_eval_job(done_incomplete, "cpu")

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
                        "--checkpoint-preview-count",
                        "3",
                        "--log-every",
                        "1",
                    ]
                )
            self.assertEqual(result["batch"]["batch_cap"], 4)
            self.assertEqual(result["batch"]["effective_batch_size"], 4)
            self.assertTrue(result["batch"]["drop_last"])
            self.assertIn("ambient_f_mem", result["final_metrics"])
            self.assertIn("latent_f_mem", result["final_metrics"])
            self.assertIn("random_t_loss", result["final_loss_eval"])
            self.assertIn("loss_t_0_5", result["final_loss_eval"])
            self.assertIn("loss_t_0_75", result["final_loss_eval"])
            self.assertNotIn("f_mem", result["final_metrics"])
            self.assertTrue(Path(result["paths"]["metrics_along_training"]).is_file())
            self.assertTrue(Path(result["paths"]["loss_eval"]).is_file())
            self.assertTrue(Path(result["paths"]["checkpoint_sample_previews"]).is_file())
            self.assertTrue(Path(result["paths"]["final_samples"]).is_file())
            self.assertGreater(len(result["checkpoint_sample_previews"]), 0)
            preview_path = Path(result["checkpoint_sample_previews"][-1]["samples_path"])
            self.assertTrue(preview_path.is_file())
            self.assertEqual(torch.load(preview_path, map_location="cpu").shape[0], 3)

    def test_notebooks_have_required_sections_and_no_outputs(self):
        recipe_sections = [
            "## 0. Knobs",
            "## 1. Imports",
            "## 2. Config Construction",
            "## 3. Data Bundle",
            "## 4. Model Construction",
            "## 5. Metric Calculation",
            "## 6. Optional Full Job",
            "## 7. Artifact Figure Loading",
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
            (ROOT / "prototyping/runners/notebooks/image_runner.ipynb", recipe_sections),
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
