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
    from mp_runners import (
        image_runner,
        synthetic_runner,
    )
except ModuleNotFoundError:
    torch = None
    build_data_bundle = None
    image_runner = None
    synthetic_runner = None


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
        (args.n_test,),
        generator=reference_generator,
    )
    reference_latent = means[reference_labels] + args.sigma_z * torch.randn(
        args.n_test,
        args.intrinsic_dim,
        generator=reference_generator,
    )
    reference = reference_latent @ basis.T
    reference = reference + args.sigma_eta * torch.randn(
        args.n_test,
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


@unittest.skipIf(torch is None, "torch and runner dependencies are required")
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
            self.assertEqual(settings["out_dir"], "outputs/runners/image_runner")
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
            self.assertEqual(config.study.dataset_key, "celeba")
            self.assertEqual(config.tracking.wandb.project, "memparam-celeba-unet")
            self.assertEqual(
                config.tracking.wandb.name,
                "celeba-unet-N128-B64-v_v-steps30000-s0",
            )
            self.assertEqual(
                config.tracking.wandb.group,
                "celeba-unet-steps30000",
            )
            self.assertEqual(config.training.ckpt_schedule, "bonnaire-1m")

    def test_image_runner_exposes_train_loss_log_cadence(self):
        settings = image_runner.default_settings()
        settings["log_every"] = 7
        config = image_runner.build_config(settings)
        args = image_runner.parse_args(["--log_every", "7"])
        default_args = image_runner.parse_args([])

        self.assertEqual(config.training.log_every, 7)
        self.assertEqual(args.log_every, 7)
        self.assertEqual(default_args.precision, "fp32")
        self.assertFalse(default_args.data_summary)

    def test_image_runner_accepts_slim_cli_surface(self):
        args = image_runner.parse_args(
            [
                "--device",
                "cpu",
                "--out_dir",
                "tmp/image32",
                "--run_name",
                "slim",
                "--seed",
                "3",
                "--dataset",
                "mnist",
                "--image_size",
                "32",
                "--model",
                "unet",
                "--config_preset",
                "image32_32",
                "--prediction_space",
                "eps",
                "--loss_space",
                "eps",
                "--n_train",
                "12",
                "--n_test",
                "10",
                "--n_training_steps",
                "5",
                "--ckpt_every",
                "2",
                "--log_every",
                "1",
                "--batch_cap",
                "4",
                "--precision",
                "bf16",
                "--data_summary",
                "--generation_monitor_samples",
                "6",
                "--sampling_steps",
                "3",
                "--root",
                "data",
                "--no_download",
                "--no_wandb",
                "--wandb_mode",
                "offline",
                "--wandb_project",
                "proj",
                "--wandb_name",
                "name",
                "--wandb_group",
                "group",
                "--wandb_tags",
                "a,b",
                "--generation_monitor",
                "--no_generation_monitor_show",
                "--random_projected_f_mem_samples",
                "7",
                "--random_projected_f_mem_resize",
                "8",
                "--random_projected_f_mem_proj_dim",
                "4",
                "--random_projected_f_mem_proj_seed",
                "23",
            ]
        )
        config = image_runner.build_config(args)

        self.assertEqual(args.prediction_space, "eps")
        self.assertEqual(args.loss_space, "eps")
        self.assertEqual(config.framework.prediction_space, "eps")
        self.assertEqual(config.framework.loss_space, "eps")
        self.assertEqual(args.image_size, 32)
        self.assertEqual(args.model, "unet")
        self.assertFalse(args.download)
        self.assertFalse(args.wandb)
        self.assertEqual(args.precision, "bf16")
        self.assertTrue(args.data_summary)
        self.assertTrue(args.generation_monitor)
        self.assertEqual(config.runtime.precision, "bf16")
        self.assertEqual(config.tracking.wandb.project, "proj")
        self.assertEqual(config.tracking.wandb.name, "name")
        self.assertEqual(config.tracking.wandb.group, "group")
        self.assertEqual(config.generation_monitor.n_samples, 6)
        self.assertFalse(config.generation_monitor.show)
        self.assertEqual(config.random_projected_f_mem.n_samples, 7)
        self.assertEqual(config.random_projected_f_mem.resize_size, 8)
        self.assertEqual(config.random_projected_f_mem.projection_dim, 4)
        self.assertEqual(config.random_projected_f_mem.projection_seed, 23)
        self.assertEqual(config.tracking.wandb.tags, ["a", "b"])

    def test_image_runner_rejects_inline_eval_flags(self):
        for flag in ("--final-eval", "--eval-run-dir"):
            with self.subTest(flag=flag):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        image_runner.parse_args([flag, "DONE_some_attempt"])

    def test_image_runner_accepts_checkpoint_schedule_flags(self):
        args = image_runner.parse_args(
            [
                "--n_training_steps",
                "1000000",
                "--ckpt_schedule",
                "bonnaire-1m",
            ]
        )
        config = image_runner.build_config(vars(args))

        self.assertEqual(config.training.ckpt_schedule, "bonnaire-1m")
        self.assertEqual(len(config.training.resolved_ckpt_steps), 72)
        self.assertEqual(config.training.resolved_ckpt_steps[0], 0)
        self.assertEqual(config.training.resolved_ckpt_steps[-1], 1000000)

    def test_image_runner_ckpt_every_flag_selects_interval_schedule(self):
        args = image_runner.parse_args(["--n_training_steps", "250", "--ckpt_every", "100"])
        config = image_runner.build_config(vars(args))

        self.assertEqual(config.training.ckpt_schedule, "interval")
        self.assertEqual(tuple(config.training.resolved_ckpt_steps), (0, 100, 200, 250))

    def test_image32_config_preset_owns_rare_geometry_and_architecture(self):
        settings = image_runner.default_settings()
        settings.update({"dataset": "flowers", "config_preset": "image32_128"})

        config = image_runner.build_config(settings)

        self.assertEqual(config.data.dataset, "flowers102")
        self.assertEqual(config.study.dataset_key, "flowers")
        self.assertEqual(config.data.train_split, "test")
        self.assertEqual(config.data.reference_split, "train")
        self.assertEqual(config.data.image_size, 128)
        self.assertEqual(config.data.resize_size, 128)
        self.assertEqual(config.data.image_channels, 3)
        self.assertEqual(config.network.base_channels, 64)
        self.assertEqual(tuple(config.network.channel_mult), (1, 2, 4, 4, 8))
        self.assertEqual(tuple(config.network.attn_resolutions), (16,))
        self.assertEqual(config.network.dropout, 0.0)

    def test_flowers_64_profile_sets_data_and_unet_shape(self):
        settings = image_runner.default_settings()
        settings.update({"dataset": "flowers", "image_size": 64})

        config = image_runner.build_config(settings)
        args = image_runner.parse_args(["--dataset", "flowers", "--image_size", "64"])

        self.assertEqual(args.dataset, "flowers")
        self.assertEqual(config.data.dataset, "flowers102")
        self.assertEqual(config.study.dataset_key, "flowers")
        self.assertEqual(config.data.image_size, 64)
        self.assertEqual(config.data.resize_size, 64)
        self.assertEqual(config.data.image_channels, 3)
        self.assertEqual(config.data.train_split, "test")
        self.assertEqual(config.data.reference_split, "train")
        self.assertEqual(config.network.base_channels, 64)
        self.assertEqual(tuple(config.network.channel_mult), (1, 2, 4, 4))
        self.assertEqual(tuple(config.network.attn_resolutions), (16,))
        self.assertEqual(config.network.dropout, 0.0)

    def test_image_runner_resolves_dataset_size_and_model_inputs(self):
        args = image_runner.parse_args(
            [
                "--dataset",
                "flowers",
                "--image_size",
                "64",
                "--model",
                "unet",
            ]
        )
        config = image_runner.build_config(vars(args))

        self.assertEqual(config.study.dataset_key, "flowers")
        self.assertEqual(config.data.dataset, "flowers102")
        self.assertEqual(config.data.image_size, 64)
        self.assertEqual(config.data.resize_size, 64)
        self.assertEqual(config.network.arch, "unet")
        self.assertEqual(config.network.base_channels, 64)
        self.assertEqual(tuple(config.network.channel_mult), (1, 2, 4, 4))

    def test_image_runner_can_resize_mnist_from_simple_inputs(self):
        args = image_runner.parse_args(
            [
                "--dataset",
                "mnist",
                "--image_size",
                "64",
                "--model",
                "unet",
            ]
        )
        config = image_runner.build_config(vars(args))

        self.assertEqual(config.study.dataset_key, "mnist")
        self.assertEqual(config.data.dataset, "mnist")
        self.assertEqual(config.data.image_size, 64)
        self.assertEqual(config.network.arch, "unet")
        self.assertEqual(config.network.base_channels, 64)
        self.assertEqual(tuple(config.network.attn_resolutions), (16,))

    def test_image32_wandb_project_uses_dataset_key_and_model(self):
        settings = image_runner.default_settings()
        settings.update({"dataset": "flowers", "image_size": 128})

        config = image_runner.build_config(settings)

        self.assertEqual(config.data.dataset, "flowers102")
        self.assertEqual(config.study.dataset_key, "flowers")
        self.assertEqual(config.network.arch, "unet")
        self.assertEqual(config.tracking.wandb.project, "memparam-flowers-unet")

    def test_image_runner_public_dataset_names_and_size_errors(self):
        valid_pairs = [
            ("mnist", 32),
            ("flowers", 64),
            ("afhq", 128),
            ("ffhq", 128),
            ("celeba", 32),
        ]
        for dataset, image_size in valid_pairs:
            with self.subTest(dataset=dataset, image_size=image_size):
                args = image_runner.parse_args(
                    ["--dataset", dataset, "--image_size", str(image_size)]
                )
                config = image_runner.build_config(vars(args))
                self.assertEqual(config.study.dataset_key, dataset)
                self.assertEqual(config.data.image_size, image_size)

        for dataset, image_size in (("ffhq", 64), ("celeba", 128)):
            with self.subTest(dataset=dataset, image_size=image_size):
                args = image_runner.parse_args(
                    ["--dataset", dataset, "--image_size", str(image_size)]
                )
                with self.assertRaisesRegex(ValueError, "only supports image_size"):
                    image_runner.build_config(vars(args))

        for dataset in ("flowers102_64", "flowers102_128", "ffhq128", "celeba32_grey", "afhq_v2"):
            with self.subTest(dataset=dataset):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        image_runner.parse_args(["--dataset", dataset])

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

        args = image_runner.parse_args(
            [
                "--dataset",
                "mnist",
                "--wandb_project",
                "memparam-image32-unet",
            ]
        )
        explicit = image_runner.build_config(vars(args))

        self.assertEqual(explicit.tracking.wandb.project, "memparam-image32-unet")

    def test_image_config_rejects_impossible_known_dataset_counts(self):
        settings = image_runner.default_settings()
        settings.update({"dataset": "flowers", "image_size": 64, "n_train": 7000})

        with self.assertRaisesRegex(ValueError, "n_train=7000 exceeds flowers102"):
            image_runner.build_config(settings)

        settings = image_runner.default_settings()
        settings.update({"dataset": "flowers", "image_size": 64, "n_test": 2000})

        with self.assertRaisesRegex(ValueError, "n_test=2000 exceeds flowers102"):
            image_runner.build_config(settings)

    def test_image_runners_accept_parameterization_spaces(self):
        for name, module, _, _ in RECIPE_RUNNERS:
            for space in ("x", "eps", "v"):
                settings = module.default_settings()
                settings.update(
                    {
                        "prediction_space": space,
                        "loss_space": space,
                        "n_train": 4,
                        "n_test": 4,
                        "n_training_steps": 2,
                        "ckpt_every": 1,
                    }
                )
                config = module.build_config(settings)
                self.assertEqual(config.framework.prediction_space, space)
                self.assertEqual(config.framework.loss_space, space)

    def test_release_clean_flags_are_removed(self):
        removed = [
            (synthetic_runner, ["--batch-size", "5"]),
            (synthetic_runner, ["--validate-only"]),
            (synthetic_runner, ["--data-source", "both"]),
            (synthetic_runner, ["--pred-space", "eps"]),
            (synthetic_runner, ["--metric-every", "1"]),
            (synthetic_runner, ["--checkpoint-preview-count", "2"]),
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
                    (module, ["--run-name", "old"]),
                    (module, ["--image-size", "32"]),
                    (module, ["--config-preset", "image32_32"]),
                    (module, ["--prediction-space", "v"]),
                    (module, ["--loss-space", "v"]),
                    (module, ["--n-eval", "4"]),
                    (module, ["--n-steps", "4"]),
                    (module, ["--eval-samples", "4"]),
                    (module, ["--generation-trace"]),
                    (module, ["--train-monitor-projected-f-mem"]),
                    (module, ["--compare-numerics"]),
                    (module, ["--n_eval", "4"]),
                    (module, ["--n_steps", "4"]),
                    (module, ["--eval_samples", "4"]),
                    (module, ["--generation_trace"]),
                    (module, ["--train_monitor_projected_f_mem"]),
                    (module, ["--compare_numerics"]),
                    (module, ["--precision", "fp16"]),
                ]
            )
        for module, argv in removed:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.parse_args(argv)

    def test_synthetic_accepts_release_clean_public_flags(self):
        args = synthetic_runner.parse_args(
            [
                "--prediction-space",
                "eps",
                "--loss-space",
                "eps",
                "--n-reference",
                "11",
                "--ambient-dim",
                "7",
                "--intrinsic-data-dim",
                "3",
                "--out-dir",
                "tmp/alias_check",
            ]
        )
        self.assertEqual(args.prediction_space, "eps")
        self.assertEqual(args.loss_space, "eps")
        self.assertEqual(args.n_test, 11)
        self.assertEqual(args.ambient_dim, 7)
        self.assertEqual(args.intrinsic_data_dim, 3)
        self.assertEqual(args.data_dim, 7)
        self.assertEqual(args.intrinsic_dim, 3)
        self.assertEqual(args.out_dir, "tmp/alias_check")
        self.assertFalse(args.final_eval)
        self.assertIsNone(args.eval_run_dir)

    def test_synthetic_runner_accepts_checkpoint_schedule_flags(self):
        args = synthetic_runner.parse_args(
            [
                "--n-steps",
                "3",
                "--ckpt-every",
                "2",
                "--ckpt-schedule",
                "bonnaire-1m",
                "--ckpt-steps",
                "0,1,3",
            ]
        )
        config = synthetic_runner.build_config(args)

        self.assertEqual(config.training.ckpt_schedule, "bonnaire-1m")
        self.assertEqual(tuple(config.training.ckpt_steps), (0, 1, 3))
        self.assertEqual(tuple(config.training.resolved_ckpt_steps), (0, 1, 3))

    def test_synthetic_runner_ckpt_every_flag_selects_interval_schedule(self):
        args = synthetic_runner.parse_args(["--n-steps", "250", "--ckpt-every", "100"])
        config = synthetic_runner.build_config(args)

        self.assertEqual(config.training.ckpt_schedule, "interval")
        self.assertEqual(tuple(config.training.resolved_ckpt_steps), (0, 100, 200, 250))

    def test_synthetic_runner_rejects_bad_checkpoint_steps(self):
        args = synthetic_runner.parse_args(["--n-steps", "3", "--ckpt-steps", "1,,2"])
        with self.assertRaisesRegex(ValueError, "comma-separated"):
            synthetic_runner.build_config(args)

        args = synthetic_runner.parse_args(["--n-steps", "3", "--ckpt-steps", "0,4"])
        with self.assertRaisesRegex(ValueError, "outside valid range"):
            synthetic_runner.build_config(args)

    def test_synthetic_torus_uses_ambient_dim_and_rejects_intrinsic_dim_flag(self):
        args = synthetic_runner.parse_args(
            [
                "--dataset",
                "torus3d",
                "--ambient-dim",
                "16",
            ]
        )
        synthetic_runner._validate_args(args)
        config = synthetic_runner.build_config(args)

        self.assertEqual(config.data.ambient_dim, 16)
        self.assertEqual(config.data.data_dim, 16)
        self.assertEqual(config.data.base_embedding_dim, 3)
        self.assertEqual(config.data.local_manifold_dim, 2)
        self.assertEqual(config.data.metric_feature_dim, 4)

        bad_args = synthetic_runner.parse_args(
            [
                "--dataset",
                "torus3d",
                "--intrinsic-data-dim",
                "2",
            ]
        )
        with self.assertRaisesRegex(ValueError, "no configurable --intrinsic-data-dim"):
            synthetic_runner._validate_args(bad_args)

    def test_synthetic_torus_defaults_to_visible_donut_dimension(self):
        args = synthetic_runner.parse_args(["--dataset", "torus3d"])
        synthetic_runner._validate_args(args)
        config = synthetic_runner.build_config(args)

        self.assertEqual(config.data.ambient_dim, 3)
        self.assertEqual(config.data.data_dim, 3)

    def test_synthetic_recipe_function_order_is_visible(self):
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
        text = Path(synthetic_runner.__file__).read_text()
        positions = [text.index(marker) for marker in expected_order]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("json.dumps", text)
        self.assertNotIn("wandb.init", text)
        self.assertNotIn("plt.subplots", text)
        self.assertIn("from mp_training import (", text)
        self.assertIn("train_run", text)
        self.assertIn("evaluate_synthetic_run", text)

    def test_image_runner_is_the_only_image_recipe_surface(self):
        text = Path(image_runner.__file__).read_text()
        self.assertIn("mp_runners.image_eval", text)
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
        args = synthetic_runner.parse_args(
            [
                "--device",
                "cpu",
                "--n-train",
                "8",
                "--n-reference",
                "12",
                "--batch-cap",
                "4",
                "--ambient-dim",
                "8",
                "--intrinsic-data-dim",
                "2",
                "--n-clusters",
                "3",
            ]
        )
        self.assertEqual(args.ambient_dim, 8)
        self.assertEqual(args.intrinsic_data_dim, 2)
        self.assertEqual(args.data_dim, 8)
        self.assertEqual(args.intrinsic_dim, 2)
        config = synthetic_runner.build_config(args)
        self.assertEqual(config.data.ambient_dim, 8)
        self.assertEqual(config.data.intrinsic_data_dim, 2)
        self.assertEqual(config.data.data_dim, 8)
        self.assertEqual(config.data.intrinsic_dim, 2)
        self.assertEqual(config.tracking.wandb.project, "memparam-gmm_subspace-mlp")
        self.assertEqual(
            config.tracking.wandb.name,
            "gmm_subspace-mlp-N8-B4-v_v-steps10000-s0",
        )
        self.assertEqual(config.tracking.wandb.group, "gmm_subspace-mlp-steps10000")
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
                "n_test": 10,
                "n_training_steps": 2,
                "ckpt_every": 1,
                "batch_cap": 2,
                "generation_monitor_samples": 2,
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
                "n_test": 4,
                "n_training_steps": 2,
                "ckpt_every": 1,
                "batch_cap": 2,
                "generation_monitor_samples": 2,
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
    def test_image_runner_default_run_keeps_generation_monitor_and_random_projected_f_mem(self):
        settings = image_runner.default_settings()
        settings.update(
            {
                "dataset": "mnist",
                "n_train": 4,
                "n_test": 4,
                "n_training_steps": 1,
                "ckpt_every": 1,
                "batch_cap": 2,
                "generation_monitor_samples": 3,
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
                "generation_monitor": True,
                "generation_monitor_show": False,
                "random_projected_f_mem_samples": 3,
                "random_projected_f_mem_resize": 8,
                "random_projected_f_mem_proj_dim": 4,
                "random_projected_f_mem_proj_seed": 23,
            }
        )
        config = image_runner.build_config(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = image_runner.run_full_job(config, tmpdir, "cpu")

            self.assertNotIn("checkpoint_metrics", result)
            self.assertNotIn("loss_eval", result)
            self.assertNotIn("checkpoint_sample_previews", result)
            self.assertFalse(Path(result["artifacts"]["checkpoint_metrics"]["path"]).exists())
            self.assertFalse(Path(result["artifacts"]["loss_eval"]["path"]).exists())
            self.assertFalse(
                Path(result["artifacts"]["checkpoint_sample_previews"]["path"]).exists()
            )
            summary_path = Path(result["run_dir"]) / "artifacts" / "run_summary.json"
            run_summary = json.loads(summary_path.read_text())
            trace = run_summary["generation_monitor"]
            self.assertTrue(Path(trace["z_init_path"]).is_file())
            self.assertTrue(Path(trace["train_grid_path"]).is_file())
            self.assertTrue(Path(trace["final_samples_path"]).is_file())
            self.assertGreater(len(trace["checkpoint_samples"]), 0)
            monitor = run_summary["random_projected_f_mem"]
            self.assertTrue(monitor["enabled"])
            self.assertEqual(monitor["metadata"]["n_samples"], 3)
            self.assertEqual(monitor["metadata"]["resize_size"], 8)
            self.assertEqual(monitor["metadata"]["projection_dim"], 4)
            self.assertTrue(
                (Path(result["run_dir"]) / monitor["metadata"]["projection_matrix_path"]).is_file()
            )
            self.assertEqual(len(monitor["records"]), 1)
            self.assertEqual(monitor["records"][0]["step"], 1)
            self.assertIn("f_mem", monitor["records"][0])
            self.assertTrue(
                (
                    Path(result["run_dir"])
                    / "metrics"
                    / "random_projected_f_mem.json"
                ).is_file()
            )

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_generation_monitor_sample_grid_is_not_capped_by_train_grid(self):
        settings = image_runner.default_settings()
        settings.update(
            {
                "dataset": "mnist",
                "n_train": 2,
                "n_test": 4,
                "n_training_steps": 1,
                "batch_cap": 2,
                "generation_monitor_samples": 4,
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
                "generation_monitor": True,
                "generation_monitor_show": False,
            }
        )
        config = image_runner.build_config(settings)
        saved_grids = []

        def fake_image_rows(rows, row_labels, figure_path, *, title=None):
            figure_path = Path(figure_path)
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            figure_path.write_bytes(b"fake")
            saved_grids.append(
                {
                    "name": figure_path.name,
                    "row_labels": list(row_labels),
                    "row_counts": [int(row.shape[0]) for row in rows],
                }
            )
            return figure_path

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("mp_training.trainer.save_image_rows_figure", fake_image_rows):
                with contextlib.redirect_stdout(io.StringIO()):
                    image_runner.run_full_job(config, tmpdir, "cpu")

        train_grid = next(
            item for item in saved_grids if item["name"] == "generation_monitor_train_grid.png"
        )
        sample_grids = [
            item
            for item in saved_grids
            if item["name"].startswith("generation_monitor_samples_")
        ]

        self.assertEqual(train_grid["row_labels"], ["train"])
        self.assertEqual(train_grid["row_counts"], [2])
        self.assertTrue(sample_grids)
        self.assertTrue(
            all(item["row_labels"] == ["generated"] for item in sample_grids)
        )
        self.assertTrue(all(item["row_counts"] == [4] for item in sample_grids))

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_image32_train_run_writes_generation_monitor_checkpoint_panels(self):
        settings = image_runner.default_settings()
        settings.update(
            {
                "dataset": "mnist",
                "n_train": 4,
                "n_test": 4,
                "n_training_steps": 1,
                "ckpt_every": 1,
                "batch_cap": 2,
                "generation_monitor_samples": 2,
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
                "generation_monitor": True,
                "generation_monitor_show": False,
            }
        )
        config = image_runner.build_config(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = image_runner.run_full_job(config, tmpdir, "cpu")

            summary_path = Path(result["run_dir"]) / "artifacts" / "run_summary.json"
            run_summary = json.loads(summary_path.read_text())
            trace = run_summary["generation_monitor"]
            self.assertTrue(Path(trace["z_init_path"]).is_file())
            self.assertTrue(Path(trace["train_grid_path"]).is_file())
            self.assertTrue(Path(trace["final_samples_path"]).is_file())
            checkpoint_trace = trace["checkpoint_samples"]
            self.assertGreater(len(checkpoint_trace), 0)
            self.assertEqual(checkpoint_trace[0]["step"], 0)
            self.assertTrue(Path(checkpoint_trace[-1]["path"]).is_file())

    def test_synthetic_tiny_full_run_writes_release_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = synthetic_runner.main(
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
                        "--ambient-dim",
                        "8",
                        "--intrinsic-data-dim",
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
                        "--ckpt-every",
                        "1",
                        "--log-every",
                        "1",
                        "--final-eval",
                    ]
                )
            self.assertEqual(result["batch"]["batch_cap"], 4)
            self.assertEqual(result["batch"]["effective_batch_size"], 4)
            self.assertTrue(result["batch"]["drop_last"])
            self.assertIn("ambient_nearest_neighbor_ratio_d1_over_d2", result["final_metrics"])
            self.assertIn("intrinsic_nearest_neighbor_ratio_d1_over_d2", result["final_metrics"])
            self.assertIn(
                "holdout_ambient_nearest_neighbor_ratio_d1_over_d2",
                result["final_metrics"],
            )
            self.assertIn(
                "holdout_intrinsic_nearest_neighbor_ratio_d1_over_d2",
                result["final_metrics"],
            )
            self.assertIn("intrinsic_sliced_w2", result["final_metrics"])
            self.assertIn("intrinsic_gmm_nll_mean", result["final_metrics"])
            self.assertIn("random_t_loss", result["final_loss_eval"])
            self.assertIn("loss_t_0_5", result["final_loss_eval"])
            self.assertIn("loss_t_0_75", result["final_loss_eval"])
            self.assertNotIn("f_mem", result["final_metrics"])
            self.assertNotIn("latent_f_mem", result["final_metrics"])
            self.assertNotIn("component_weight_l1", result["final_metrics"])
            self.assertNotIn("oracle_full_nll_mean", result["final_metrics"])
            self.assertTrue(Path(result["paths"]["metrics_along_training"]).is_file())
            self.assertTrue(Path(result["paths"]["loss_eval"]).is_file())
            self.assertTrue(Path(result["paths"]["checkpoint_sample_previews"]).is_file())
            self.assertTrue(Path(result["paths"]["final_samples"]).is_file())
            self.assertGreater(len(result["checkpoint_sample_previews"]), 0)
            preview_path = Path(result["checkpoint_sample_previews"][-1]["samples_path"])
            self.assertTrue(preview_path.is_file())
            self.assertEqual(torch.load(preview_path, map_location="cpu").shape[0], 4)

    def test_synthetic_default_run_skips_final_eval(self):
        settings = synthetic_runner.default_settings()
        settings.update(
            {
                "n_train": 6,
                "n_test": 8,
                "batch_cap": 3,
                "eval_samples": 3,
                "sampling_steps": 2,
                "ambient_dim": 4,
                "intrinsic_data_dim": 2,
                "n_clusters": 2,
                "hidden_dim": 16,
                "n_layers": 2,
                "time_emb_dim": 16,
                "n_steps": 1,
                "ckpt_every": 1,
                "log_every": 1,
                "wandb": False,
            }
        )
        config = synthetic_runner.build_config(settings)
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = synthetic_runner.run_full_job(config, tmpdir, "cpu")

            self.assertEqual(result["checkpoint_metrics"], [])
            self.assertEqual(result["loss_eval"], [])
            self.assertEqual(result["checkpoint_sample_previews"], [])
            self.assertTrue(Path(result["paths"]["train_history"]).is_file())
            self.assertFalse(Path(result["paths"]["checkpoint_metrics"]).exists())
            self.assertTrue(Path(result["run_dir"]).name.startswith("DONE_"))

    def test_synthetic_eval_run_dir_replays_finished_done_run(self):
        settings = synthetic_runner.default_settings()
        settings.update(
            {
                "n_train": 6,
                "n_test": 8,
                "batch_cap": 3,
                "eval_samples": 3,
                "sampling_steps": 2,
                "data_dim": 4,
                "intrinsic_dim": 2,
                "n_clusters": 2,
                "hidden_dim": 16,
                "n_layers": 2,
                "time_emb_dim": 16,
                "n_steps": 1,
                "ckpt_every": 1,
                "log_every": 1,
                "wandb": False,
            }
        )
        config = synthetic_runner.build_config(settings)
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                train_result = synthetic_runner.run_full_job(config, tmpdir, "cpu")
            run_dir = Path(train_result["run_dir"])

            with contextlib.redirect_stdout(io.StringIO()):
                eval_result = synthetic_runner.main(
                    ["--device", "cpu", "--eval-run-dir", str(run_dir)]
                )

            self.assertEqual(eval_result["mode"], "eval_run_dir")
            self.assertEqual(Path(eval_result["run_dir"]), run_dir)
            self.assertGreater(len(eval_result["checkpoint_metrics"]), 0)
            self.assertIn(
                "intrinsic_nearest_neighbor_ratio_d1_over_d2",
                eval_result["final_metrics"],
            )
            self.assertTrue(Path(eval_result["artifacts"]["checkpoint_metrics"]["path"]).is_file())

    def test_synthetic_eval_run_dir_rejects_unfinished_or_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "DONE_"):
                synthetic_runner.run_eval_job(root / "attempt_0001", "cpu")

            done_bad = root / "DONE_failed"
            done_bad.mkdir()
            (done_bad / "status.json").write_text(json.dumps({"state": "failed"}))
            with self.assertRaisesRegex(ValueError, "trained"):
                synthetic_runner.run_eval_job(done_bad, "cpu")

            done_incomplete = root / "DONE_incomplete"
            done_incomplete.mkdir()
            (done_incomplete / "status.json").write_text(json.dumps({"state": "trained"}))
            with self.assertRaisesRegex(FileNotFoundError, "required saved-run artifacts"):
                synthetic_runner.run_eval_job(done_incomplete, "cpu")

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
            (ROOT / "prototyping/runners/notebooks/synthetic_runner.ipynb", synthetic_sections),
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
