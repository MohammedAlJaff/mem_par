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
        eval_synthetic,
        image_train_runner,
        synthetic_train_runner,
    )
    from mp_runners.eval_safety import eval_z_init_bank, validate_saved_train_reference
except ModuleNotFoundError:
    torch = None
    build_data_bundle = None
    eval_synthetic = None
    image_train_runner = None
    synthetic_train_runner = None
    eval_z_init_bank = None
    validate_saved_train_reference = None


ROOT = Path(__file__).resolve().parents[1]
RECIPE_RUNNERS = (
    ("image_train_runner", image_train_runner, "celeba32_grey", "unet"),
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
        torch.randn(
            args.ambient_dim,
            args.intrinsic_data_dim,
            generator=geometry_generator,
        )
    )
    basis = basis[:, : args.intrinsic_data_dim]
    means = torch.randn(
        args.n_clusters,
        args.intrinsic_data_dim,
        generator=geometry_generator,
    )

    train_generator = torch.Generator().manual_seed(int(args.seed))
    train_labels = torch.randint(args.n_clusters, (args.n_train,), generator=train_generator)
    train_latent = means[train_labels] + args.sigma_z * torch.randn(
        args.n_train,
        args.intrinsic_data_dim,
        generator=train_generator,
    )
    train = train_latent @ basis.T
    train = train + args.sigma_eta * torch.randn(
        args.n_train,
        args.ambient_dim,
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
        args.intrinsic_data_dim,
        generator=reference_generator,
    )
    reference = reference_latent @ basis.T
    reference = reference + args.sigma_eta * torch.randn(
        args.n_test,
        args.ambient_dim,
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
            self.assertEqual(settings["out_dir"], "outputs/runners/image_train_runner")
            self.assertEqual(config.study.name, "image_train_runner")
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
            self.assertEqual(config.training.weight_decay, 0.0)
            self.assertEqual(config.training.momentum, 0.9)
            self.assertEqual(config.runtime.precision, "fp32")
            self.assertEqual(config.study.dataset_key, "celeba")
            self.assertEqual(config.monitoring.batch_size, settings["batch_cap"])
            self.assertEqual(config.tracking.wandb.project, "memparam-celeba-32-unet")
            self.assertEqual(
                config.tracking.wandb.name,
                "celeba-32-unet-N128-B64-v_v-steps30000-s0",
            )
            self.assertEqual(
                config.tracking.wandb.group,
                "celeba-32-unet-steps30000",
            )
            self.assertEqual(config.training.ckpt_schedule, "bonnaire-1m")

    def test_image_runner_accepts_explicit_sgd_momentum_settings(self):
        settings = image_train_runner.default_settings()
        settings.update(
            {
                "optimizer": "sgd",
                "lr": 1.0e-2,
                "weight_decay": 0.0,
                "momentum": 0.95,
            }
        )

        config = image_train_runner.build_config(settings)

        self.assertEqual(config.training.optimizer, "sgd")
        self.assertEqual(config.training.lr, 1.0e-2)
        self.assertEqual(config.training.weight_decay, 0.0)
        self.assertEqual(config.training.momentum, 0.95)

    def test_image_runner_cli_exposes_optimizer_settings(self):
        args = image_train_runner.parse_args(
            [
                "--optimizer",
                "sgd",
                "--lr",
                "0.01",
                "--weight_decay",
                "0.0",
                "--momentum",
                "0.95",
            ]
        )

        self.assertEqual(args.optimizer, "sgd")
        self.assertEqual(args.lr, 0.01)
        self.assertEqual(args.weight_decay, 0.0)
        self.assertEqual(args.momentum, 0.95)

    def test_image_runner_exposes_train_loss_log_cadence(self):
        settings = image_train_runner.default_settings()
        settings["log_every"] = 7
        config = image_train_runner.build_config(settings)
        args = image_train_runner.parse_args(["--log_every", "7"])
        default_args = image_train_runner.parse_args([])

        self.assertEqual(config.training.log_every, 7)
        self.assertEqual(args.log_every, 7)
        self.assertEqual(default_args.precision, "fp32")
        self.assertFalse(default_args.data_summary)

    def test_image_monitor_batch_size_must_be_positive(self):
        settings = image_train_runner.default_settings()
        settings["monitor_batch_size"] = 0

        with self.assertRaisesRegex(
            ValueError,
            "monitor_batch_size must be >= 1",
        ):
            image_train_runner.build_config(settings)

    def test_image_runner_accepts_slim_cli_surface(self):
        args = image_train_runner.parse_args(
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
                "--eval_sample_bank_size",
                "9",
                "--generation_monitor_samples",
                "6",
                "--monitor_batch_size",
                "11",
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
        config = image_train_runner.build_config(args)

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
        self.assertEqual(config.evaluation.sample_bank_size, 9)
        self.assertEqual(config.evaluation.n_samples, 9)
        self.assertEqual(config.generation_monitor.n_samples, 6)
        self.assertEqual(args.monitor_batch_size, 11)
        self.assertEqual(config.monitoring.batch_size, 11)
        self.assertEqual(config.training.batch_cap, 4)
        self.assertFalse(config.generation_monitor.show)
        self.assertEqual(config.random_projected_f_mem.n_samples, 7)
        self.assertEqual(config.random_projected_f_mem.resize_size, 8)
        self.assertEqual(config.random_projected_f_mem.projection_dim, 4)
        self.assertEqual(config.random_projected_f_mem.projection_seed, 23)
        self.assertEqual(config.tracking.wandb.tags, ["a", "b"])

    def test_image_runner_rejects_inline_eval_flags(self):
        for flag in ("", "--run_dir"):
            with self.subTest(flag=flag):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        image_train_runner.parse_args([flag, "DONE_some_attempt"])

    def test_image_runner_accepts_checkpoint_schedule_flags(self):
        args = image_train_runner.parse_args(
            [
                "--n_training_steps",
                "1000000",
                "--ckpt_schedule",
                "bonnaire-1m",
            ]
        )
        config = image_train_runner.build_config(vars(args))

        self.assertEqual(config.training.ckpt_schedule, "bonnaire-1m")
        self.assertEqual(len(config.training.resolved_ckpt_steps), 72)
        self.assertEqual(config.training.resolved_ckpt_steps[0], 0)
        self.assertEqual(config.training.resolved_ckpt_steps[-1], 1000000)

    def test_image_runner_ckpt_every_flag_selects_interval_schedule(self):
        args = image_train_runner.parse_args(["--n_training_steps", "250", "--ckpt_every", "100"])
        config = image_train_runner.build_config(vars(args))

        self.assertEqual(config.training.ckpt_schedule, "interval")
        self.assertEqual(tuple(config.training.resolved_ckpt_steps), (0, 100, 200, 250))

    def test_image_size_profile_owns_data_and_architecture(self):
        settings = image_train_runner.default_settings()
        settings.update({"dataset": "flowers", "image_size": 128})

        config = image_train_runner.build_config(settings)

        self.assertEqual(config.data.dataset, "flowers102")
        self.assertEqual(config.study.dataset_key, "flowers")
        self.assertEqual(config.data.recipe, "default")
        self.assertEqual(config.data.train_split, "test")
        self.assertEqual(config.data.reference_split, "train")
        self.assertEqual(config.data.image_size, 128)
        self.assertEqual(config.data.resize_size, 128)
        self.assertEqual(config.data.image_channels, 3)
        self.assertEqual(config.network.base_channels, 64)
        self.assertEqual(tuple(config.network.channel_mult), (1, 2, 4, 4, 8))
        self.assertEqual(tuple(config.network.attn_resolutions), (16,))
        self.assertEqual(config.network.dropout, 0.0)

    def test_celeba32_defaults_to_bonnaire_recipe_without_public_preset(self):
        settings = image_train_runner.default_settings()
        settings.update({"dataset": "celeba", "image_size": 32})
        config = image_train_runner.build_config(settings)

        self.assertEqual(config.data.dataset, "celeba32_grey")
        self.assertEqual(config.study.dataset_key, "celeba")
        self.assertEqual(config.data.recipe, "bonnaire_celeba32")
        self.assertEqual(config.training.ckpt_schedule, "bonnaire-1m")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                image_train_runner.parse_args(["--config_preset", "image32_32"])

    def test_flowers_64_profile_sets_data_and_unet_shape(self):
        settings = image_train_runner.default_settings()
        settings.update({"dataset": "flowers", "image_size": 64})

        config = image_train_runner.build_config(settings)
        args = image_train_runner.parse_args(["--dataset", "flowers", "--image_size", "64"])

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
        args = image_train_runner.parse_args(
            [
                "--dataset",
                "flowers",
                "--image_size",
                "64",
                "--model",
                "unet",
            ]
        )
        config = image_train_runner.build_config(vars(args))

        self.assertEqual(config.study.dataset_key, "flowers")
        self.assertEqual(config.data.dataset, "flowers102")
        self.assertEqual(config.data.image_size, 64)
        self.assertEqual(config.data.resize_size, 64)
        self.assertEqual(config.network.arch, "unet")
        self.assertEqual(config.network.base_channels, 64)
        self.assertEqual(tuple(config.network.channel_mult), (1, 2, 4, 4))

    def test_image_runner_accepts_unet_capacity_overrides(self):
        args = image_train_runner.parse_args(
            [
                "--dataset",
                "celeba",
                "--image_size",
                "32",
                "--model",
                "unet",
                "--base_channels",
                "64",
                "--channel_mult",
                "1,2,4,4",
                "--attn_resolutions",
                "16,8",
                "--num_res_blocks",
                "2",
                "--dropout",
                "0.1",
            ]
        )
        config = image_train_runner.build_config(vars(args))

        self.assertEqual(config.data.dataset, "celeba32_grey")
        self.assertEqual(config.network.base_channels, 64)
        self.assertEqual(tuple(config.network.channel_mult), (1, 2, 4, 4))
        self.assertEqual(tuple(config.network.attn_resolutions), (16, 8))
        self.assertEqual(config.network.num_res_blocks, 2)
        self.assertAlmostEqual(config.network.dropout, 0.1)

    def test_image_runner_can_resize_mnist_from_simple_inputs(self):
        args = image_train_runner.parse_args(
            [
                "--dataset",
                "mnist",
                "--image_size",
                "64",
                "--model",
                "unet",
            ]
        )
        config = image_train_runner.build_config(vars(args))

        self.assertEqual(config.study.dataset_key, "mnist")
        self.assertEqual(config.data.dataset, "mnist")
        self.assertEqual(config.data.image_size, 64)
        self.assertEqual(config.network.arch, "unet")
        self.assertEqual(config.network.base_channels, 64)
        self.assertEqual(tuple(config.network.attn_resolutions), (16,))

    def test_image32_wandb_project_uses_dataset_key_and_model(self):
        settings = image_train_runner.default_settings()
        settings.update({"dataset": "flowers", "image_size": 128})

        config = image_train_runner.build_config(settings)

        self.assertEqual(config.data.dataset, "flowers102")
        self.assertEqual(config.study.dataset_key, "flowers")
        self.assertEqual(config.network.arch, "unet")
        self.assertEqual(config.tracking.wandb.project, "memparam-flowers-128-unet")

    def test_image_runner_public_dataset_names_and_size_errors(self):
        valid_pairs = [
            ("mnist", 32),
            ("flowers", 64),
            ("afhq", 128),
            ("ffhq", 128),
            ("ffhq", 64),
            ("celeba", 32),
        ]
        for dataset, image_size in valid_pairs:
            with self.subTest(dataset=dataset, image_size=image_size):
                args = image_train_runner.parse_args(
                    ["--dataset", dataset, "--image_size", str(image_size)]
                )
                config = image_train_runner.build_config(vars(args))
                self.assertEqual(config.study.dataset_key, dataset)
                self.assertEqual(config.data.image_size, image_size)

        for dataset, image_size in (("celeba", 128),):
            with self.subTest(dataset=dataset, image_size=image_size):
                args = image_train_runner.parse_args(
                    ["--dataset", dataset, "--image_size", str(image_size)]
                )
                with self.assertRaisesRegex(ValueError, "only supports image_size"):
                    image_train_runner.build_config(vars(args))

        for dataset in ("flowers102_64", "flowers102_128", "ffhq128", "celeba32_grey", "afhq_v2"):
            with self.subTest(dataset=dataset):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        image_train_runner.parse_args(["--dataset", dataset])

    def test_image32_legacy_broad_wandb_project_routes_to_dataset_project(self):
        settings = image_train_runner.default_settings()
        settings.update(
            {
                "dataset": "mnist",
                "wandb_project": "memparam-image32-unet",
            }
        )

        config = image_train_runner.build_config(settings)

        self.assertEqual(config.tracking.wandb.project, "memparam-mnist-32-unet")

        args = image_train_runner.parse_args(
            [
                "--dataset",
                "mnist",
                "--wandb_project",
                "memparam-image32-unet",
            ]
        )
        explicit = image_train_runner.build_config(vars(args))

        self.assertEqual(explicit.tracking.wandb.project, "memparam-image32-unet")

    def test_image_config_rejects_impossible_known_dataset_counts(self):
        settings = image_train_runner.default_settings()
        settings.update({"dataset": "flowers", "image_size": 64, "n_train": 7000})

        with self.assertRaisesRegex(ValueError, "n_train=7000 exceeds flowers102"):
            image_train_runner.build_config(settings)

        settings = image_train_runner.default_settings()
        settings.update({"dataset": "flowers", "image_size": 64, "n_test": 2000})

        with self.assertRaisesRegex(ValueError, "n_test=2000 exceeds flowers102"):
            image_train_runner.build_config(settings)

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
            (synthetic_train_runner, ["--batch-size", "5"]),
            (synthetic_train_runner, ["--validate-only"]),
            (synthetic_train_runner, ["--data-source", "both"]),
            (synthetic_train_runner, ["--pred-space", "eps"]),
            (synthetic_train_runner, ["--metric-every", "1"]),
            (synthetic_train_runner, ["--checkpoint-preview-count", "2"]),
            (synthetic_train_runner, ["--out-dir", "tmp/old"]),
            (synthetic_train_runner, ["--run-name", "old"]),
            (synthetic_train_runner, ["--prediction-space", "eps"]),
            (synthetic_train_runner, ["--loss-space", "eps"]),
            (synthetic_train_runner, ["--n-train", "4"]),
            (synthetic_train_runner, ["--n-test", "4"]),
            (synthetic_train_runner, ["--n-steps", "4"]),
            (synthetic_train_runner, ["--eval-samples", "4"]),
            (synthetic_train_runner, ["--final-eval"]),
            (synthetic_train_runner, ["--eval-run-dir", "DONE_old"]),
            (synthetic_train_runner, ["--n_steps", "4"]),
            (synthetic_train_runner, ["--eval_samples", "4"]),
            (synthetic_train_runner, ["--final_eval"]),
            (synthetic_train_runner, ["--eval_run_dir", "DONE_old"]),
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
                    (module, ["--config_preset", "image32_32"]),
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
                    (module, ["--wandb_log_figures"]),
                    (module, ["--no_wandb_log_figures"]),
                    (module, ["--precision", "fp16"]),
                ]
            )
        for module, argv in removed:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.parse_args(argv)

    def test_synthetic_accepts_release_clean_public_flags(self):
        args = synthetic_train_runner.parse_args(
            [
                "--prediction_space",
                "eps",
                "--loss_space",
                "eps",
                "--n_test",
                "11",
                "--ambient_dim",
                "7",
                "--intrinsic_data_dim",
                "3",
                "--out_dir",
                "tmp/alias_check",
            ]
        )
        self.assertEqual(args.prediction_space, "eps")
        self.assertEqual(args.loss_space, "eps")
        self.assertEqual(args.n_test, 11)
        self.assertEqual(args.ambient_dim, 7)
        self.assertEqual(args.intrinsic_data_dim, 3)
        self.assertEqual(args.out_dir, "tmp/alias_check")

    def test_synthetic_runner_accepts_checkpoint_schedule_flags(self):
        args = synthetic_train_runner.parse_args(
            [
                "--n_training_steps",
                "3",
                "--ckpt_every",
                "2",
                "--ckpt_schedule",
                "bonnaire-1m",
                "--ckpt_steps",
                "0,1,3",
            ]
        )
        config = synthetic_train_runner.build_config(args)

        self.assertEqual(config.training.ckpt_schedule, "bonnaire-1m")
        self.assertEqual(tuple(config.training.ckpt_steps), (0, 1, 3))
        self.assertEqual(tuple(config.training.resolved_ckpt_steps), (0, 1, 3))

    def test_synthetic_runner_ckpt_every_flag_selects_interval_schedule(self):
        args = synthetic_train_runner.parse_args(
            ["--n_training_steps", "250", "--ckpt_every", "100"]
        )
        config = synthetic_train_runner.build_config(args)

        self.assertEqual(config.training.ckpt_schedule, "interval")
        self.assertEqual(tuple(config.training.resolved_ckpt_steps), (0, 100, 200, 250))

    def test_synthetic_runner_rejects_bad_checkpoint_steps(self):
        args = synthetic_train_runner.parse_args(
            ["--n_training_steps", "3", "--ckpt_steps", "1,,2"]
        )
        with self.assertRaisesRegex(ValueError, "comma-separated"):
            synthetic_train_runner.build_config(args)

        args = synthetic_train_runner.parse_args(
            ["--n_training_steps", "3", "--ckpt_steps", "0,4"]
        )
        with self.assertRaisesRegex(ValueError, "outside valid range"):
            synthetic_train_runner.build_config(args)

    def test_synthetic_torus_uses_ambient_dim_and_rejects_intrinsic_dim_flag(self):
        args = synthetic_train_runner.parse_args(
            [
                "--dataset",
                "torus3d",
                "--ambient_dim",
                "16",
            ]
        )
        synthetic_train_runner._validate_args(args)
        config = synthetic_train_runner.build_config(args)

        self.assertEqual(config.data.ambient_dim, 16)
        self.assertEqual(config.data.data_dim, 16)
        self.assertEqual(config.data.base_embedding_dim, 3)
        self.assertEqual(config.data.local_manifold_dim, 2)
        self.assertEqual(config.data.metric_feature_dim, 4)

        bad_args = synthetic_train_runner.parse_args(
            [
                "--dataset",
                "torus3d",
                "--intrinsic_data_dim",
                "2",
            ]
        )
        with self.assertRaisesRegex(ValueError, "no configurable --intrinsic_data_dim"):
            synthetic_train_runner._validate_args(bad_args)

    def test_synthetic_torus_defaults_to_visible_donut_dimension(self):
        args = synthetic_train_runner.parse_args(["--dataset", "torus3d"])
        synthetic_train_runner._validate_args(args)
        config = synthetic_train_runner.build_config(args)

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
        text = Path(synthetic_train_runner.__file__).read_text()
        positions = [text.index(marker) for marker in expected_order]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("json.dumps", text)
        self.assertNotIn("wandb.init", text)
        self.assertNotIn("plt.subplots", text)
        self.assertIn("from mp_training import (", text)
        self.assertIn("train_run", text)
        self.assertNotIn("evaluate_synthetic_run", text)

    def test_image_runner_is_the_only_image_recipe_surface(self):
        text = Path(image_train_runner.__file__).read_text()
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
            ROOT / "prototyping/runners/notebooks/image_train_runner.ipynb",
        ]
        for path in docs:
            text = path.read_text()
            self.assertNotIn("prototyping.runners.image32_unet", text)
            self.assertNotIn("notebooks/image32_unet.ipynb", text)
            self.assertNotIn("outputs/prototype_runners/image32_unet", text)

    def test_manual_gmm_equations_match_canonical_builder(self):
        args = synthetic_train_runner.parse_args(
            [
                "--device",
                "cpu",
                "--n_train",
                "8",
                "--n_test",
                "12",
                "--batch_cap",
                "4",
                "--ambient_dim",
                "8",
                "--intrinsic_data_dim",
                "2",
                "--n_clusters",
                "3",
            ]
        )
        self.assertEqual(args.ambient_dim, 8)
        self.assertEqual(args.intrinsic_data_dim, 2)
        config = synthetic_train_runner.build_config(args)
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
        settings = image_train_runner.default_settings()
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
        config = image_train_runner.build_config(settings)
        data = image_train_runner.build_data(config)
        model = image_train_runner.build_model(config, "cpu")
        metrics = image_train_runner.inspect_metrics(config, data, model, "cpu")

        self.assertEqual(list(data["train"].shape), [10, 1, 16, 16])
        self.assertEqual(model.net.__class__.__name__, "BonnaireUNet")
        self.assertIn("f_mem", metrics["reference_memory"])
        self.assertIn("random_t_loss", metrics["loss_eval"])
        self.assertIn("loss_t_0_5", metrics["loss_eval"])
        self.assertIn("loss_t_0_75", metrics["loss_eval"])

    @mock.patch("mp_data.celeba32.datasets.CelebA", FakeCelebA)
    def test_image_runner_recipe_steps_use_fake_celeba_dataset(self):
        settings = image_train_runner.default_settings()
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
        config = image_train_runner.build_config(settings)
        data = image_train_runner.build_data(config)
        model = image_train_runner.build_model(config, "cpu")
        metrics = image_train_runner.inspect_metrics(config, data, model, "cpu")

        self.assertEqual(list(data["train"].shape), [4, 1, 32, 32])
        self.assertEqual(model.net.__class__.__name__, "BonnaireUNet")
        self.assertIn("f_mem", metrics["reference_memory"])

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_image_runner_default_run_keeps_generation_monitor_and_random_projected_f_mem(self):
        settings = image_train_runner.default_settings()
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
        config = image_train_runner.build_config(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = image_train_runner.run_full_job(config, tmpdir, "cpu")

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
            self.assertIn("total_elapsed_sec", trace["checkpoint_samples"][0])
            monitor = run_summary["random_projected_f_mem"]
            self.assertTrue(monitor["enabled"])
            self.assertEqual(monitor["metadata"]["n_samples"], 3)
            self.assertEqual(monitor["metadata"]["resize_size"], 8)
            self.assertEqual(monitor["metadata"]["projection_dim"], 4)
            self.assertIn(
                "reference_projection_elapsed_sec",
                monitor["metadata"],
            )
            self.assertTrue(
                (Path(result["run_dir"]) / monitor["metadata"]["projection_matrix_path"]).is_file()
            )
            self.assertTrue(
                (Path(result["run_dir"]) / monitor["metadata"]["reference_vectors_path"]).is_file()
            )
            self.assertTrue(monitor["metadata"]["projected_reference_cached"])
            self.assertEqual(monitor["metadata"]["reference_vectors_shape"], [4, 4])
            self.assertEqual(len(monitor["records"]), 1)
            self.assertEqual(monitor["records"][0]["step"], 1)
            self.assertIn("f_mem", monitor["records"][0])
            self.assertIn("std_r", monitor["records"][0])
            self.assertIn("sampling_elapsed_sec", monitor["records"][0])
            self.assertIn("metric_elapsed_sec", monitor["records"][0])
            self.assertIn("total_elapsed_sec", monitor["records"][0])
            self.assertTrue(monitor["records"][0]["projected_reference_cached"])
            self.assertTrue(
                (
                    Path(result["run_dir"])
                    / "metrics"
                    / "random_projected_f_mem.json"
                ).is_file()
            )

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_image_monitor_batch_size_can_exceed_training_batch_cap(self):
        settings = image_train_runner.default_settings()
        settings.update(
            {
                "dataset": "mnist",
                "n_train": 4,
                "n_test": 4,
                "n_training_steps": 1,
                "ckpt_every": 1,
                "batch_cap": 2,
                "monitor_batch_size": 4,
                "eval_sample_bank_size": 4,
                "generation_monitor_samples": 4,
                "sampling_steps": 1,
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
                "random_projected_f_mem_samples": 4,
                "random_projected_f_mem_resize": 8,
                "random_projected_f_mem_proj_dim": 4,
                "random_projected_f_mem_proj_seed": 23,
            }
        )
        config = image_train_runner.build_config(settings)
        sample_batch_sizes = []

        def fake_sample(self, z, *args, **kwargs):
            sample_batch_sizes.append(int(z.shape[0]))
            return z

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("mp_training.trainer.GDFM.sample", fake_sample):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = image_train_runner.run_full_job(config, tmpdir, "cpu")

            run_summary = json.loads(
                (Path(result["run_dir"]) / "artifacts" / "run_summary.json").read_text()
            )

        self.assertEqual(config.training.batch_cap, 2)
        self.assertEqual(config.monitoring.batch_size, 4)
        self.assertIn(4, sample_batch_sizes)
        self.assertTrue(all(size <= 4 for size in sample_batch_sizes))
        self.assertEqual(run_summary["generation_monitor"]["monitor_batch_size"], 4)
        self.assertEqual(
            run_summary["random_projected_f_mem"]["metadata"]["monitor_batch_size"],
            4,
        )

    @mock.patch("mp_data.mnist.datasets.MNIST", FakeMNIST)
    def test_generation_monitor_sample_grid_is_not_capped_by_train_grid(self):
        settings = image_train_runner.default_settings()
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
        config = image_train_runner.build_config(settings)
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
                    image_train_runner.run_full_job(config, tmpdir, "cpu")

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
        settings = image_train_runner.default_settings()
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
        config = image_train_runner.build_config(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = image_train_runner.run_full_job(config, tmpdir, "cpu")

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

    def test_synthetic_tiny_train_run_then_eval_writes_release_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = synthetic_train_runner.main(
                    [
                        "--device",
                        "cpu",
                        "--out_dir",
                        tmpdir,
                        "--run_name",
                        "tiny_synth",
                        "--n_train",
                        "8",
                        "--n_test",
                        "12",
                        "--batch_cap",
                        "4",
                        "--eval_sample_bank_size",
                        "4",
                        "--sampling_steps",
                        "2",
                        "--ambient_dim",
                        "8",
                        "--intrinsic_data_dim",
                        "2",
                        "--n_clusters",
                        "3",
                        "--hidden_dim",
                        "16",
                        "--n_layers",
                        "2",
                        "--time_emb_dim",
                        "16",
                        "--sliced_w2_projections",
                        "8",
                        "--n_training_steps",
                        "2",
                        "--ckpt_every",
                        "1",
                        "--log_every",
                        "1",
                        "--no_wandb",
                    ]
                )
            self.assertEqual(result["batch"]["batch_cap"], 4)
            self.assertEqual(result["batch"]["effective_batch_size"], 4)
            self.assertTrue(result["batch"]["drop_last"])
            self.assertNotIn("final_metrics", result)
            self.assertTrue(Path(result["paths"]["train_history"]).is_file())
            self.assertFalse(Path(result["paths"]["metrics_along_training"]).exists())

            eval_result = eval_synthetic.run_eval_job(result["run_dir"], "cpu")
            self.assertIn(
                "ambient_nearest_neighbor_ratio_d1_over_d2",
                eval_result["final_metrics"],
            )
            self.assertIn(
                "intrinsic_nearest_neighbor_ratio_d1_over_d2",
                eval_result["final_metrics"],
            )
            self.assertIn(
                "holdout_ambient_nearest_neighbor_ratio_d1_over_d2",
                eval_result["final_metrics"],
            )
            self.assertIn(
                "holdout_intrinsic_nearest_neighbor_ratio_d1_over_d2",
                eval_result["final_metrics"],
            )
            self.assertIn("intrinsic_sliced_w2", eval_result["final_metrics"])
            self.assertIn("intrinsic_gmm_nll_mean", eval_result["final_metrics"])
            self.assertIn("random_t_loss", eval_result["final_loss_eval"])
            self.assertIn("loss_t_0_5", eval_result["final_loss_eval"])
            self.assertIn("loss_t_0_75", eval_result["final_loss_eval"])
            self.assertNotIn("f_mem", eval_result["final_metrics"])
            self.assertNotIn("latent_f_mem", eval_result["final_metrics"])
            self.assertNotIn("component_weight_l1", eval_result["final_metrics"])
            self.assertNotIn("oracle_full_nll_mean", eval_result["final_metrics"])
            self.assertTrue(Path(eval_result["artifacts"]["checkpoint_metrics"]["path"]).is_file())
            self.assertTrue(Path(eval_result["artifacts"]["loss_eval"]["path"]).is_file())
            self.assertTrue(
                Path(eval_result["artifacts"]["checkpoint_sample_previews"]["path"]).is_file()
            )
            self.assertTrue(Path(eval_result["artifacts"]["final_samples"]["path"]).is_file())
            self.assertGreater(len(eval_result["checkpoint_sample_previews"]), 0)
            preview_path = Path(eval_result["checkpoint_sample_previews"][-1]["samples_path"])
            self.assertTrue(preview_path.is_file())
            self.assertEqual(torch.load(preview_path, map_location="cpu").shape[0], 4)

    def test_synthetic_default_run_saves_train_artifacts_only(self):
        settings = synthetic_train_runner.default_settings()
        settings.update(
            {
                "n_train": 6,
                "n_test": 8,
                "batch_cap": 3,
                "eval_sample_bank_size": 3,
                "sampling_steps": 2,
                "ambient_dim": 4,
                "intrinsic_data_dim": 2,
                "n_clusters": 2,
                "hidden_dim": 16,
                "n_layers": 2,
                "time_emb_dim": 16,
                "n_training_steps": 1,
                "ckpt_every": 1,
                "log_every": 1,
                "wandb": False,
            }
        )
        config = synthetic_train_runner.build_config(settings)
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = synthetic_train_runner.run_full_job(config, tmpdir, "cpu")

            self.assertNotIn("checkpoint_metrics", result)
            self.assertNotIn("loss_eval", result)
            self.assertNotIn("checkpoint_sample_previews", result)
            self.assertTrue(Path(result["paths"]["train_history"]).is_file())
            self.assertFalse(Path(result["paths"]["metrics_along_training"]).exists())
            self.assertTrue(Path(result["run_dir"]).name.startswith("DONE_"))

    def test_synthetic_run_dir_replays_finished_done_run(self):
        settings = synthetic_train_runner.default_settings()
        settings.update(
            {
                "n_train": 6,
                "n_test": 8,
                "batch_cap": 3,
                "eval_sample_bank_size": 3,
                "sampling_steps": 2,
                "ambient_dim": 4,
                "intrinsic_data_dim": 2,
                "n_clusters": 2,
                "hidden_dim": 16,
                "n_layers": 2,
                "time_emb_dim": 16,
                "n_training_steps": 1,
                "ckpt_every": 1,
                "log_every": 1,
                "wandb": False,
            }
        )
        config = synthetic_train_runner.build_config(settings)
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                train_result = synthetic_train_runner.run_full_job(config, tmpdir, "cpu")
            run_dir = Path(train_result["run_dir"])

            with contextlib.redirect_stdout(io.StringIO()):
                eval_result = eval_synthetic.main(["--device", "cpu", "--run_dir", str(run_dir)])

            self.assertEqual(eval_result["mode"], "run_dir")
            self.assertEqual(Path(eval_result["run_dir"]), run_dir)
            self.assertGreater(len(eval_result["checkpoint_metrics"]), 0)
            self.assertIn(
                "intrinsic_nearest_neighbor_ratio_d1_over_d2",
                eval_result["final_metrics"],
            )
            self.assertTrue(Path(eval_result["artifacts"]["checkpoint_metrics"]["path"]).is_file())

    def test_synthetic_run_dir_rejects_unfinished_or_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "DONE_"):
                eval_synthetic.run_eval_job(root / "attempt_0001", "cpu")

            done_bad = root / "DONE_failed"
            done_bad.mkdir()
            (done_bad / "status.json").write_text(json.dumps({"state": "failed"}))
            with self.assertRaisesRegex(ValueError, "trained"):
                eval_synthetic.run_eval_job(done_bad, "cpu")

            done_incomplete = root / "DONE_incomplete"
            done_incomplete.mkdir()
            (done_incomplete / "status.json").write_text(json.dumps({"state": "trained"}))
            with self.assertRaisesRegex(FileNotFoundError, "required saved-run artifacts"):
                eval_synthetic.run_eval_job(done_incomplete, "cpu")

    def test_saved_run_eval_safety_rejects_overlap_and_short_noise_bank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            artifacts = run_dir / "artifacts"
            artifacts.mkdir()
            (artifacts / "data_meta.json").write_text(
                json.dumps(
                    {
                        "train_source_id": "same",
                        "reference_source_id": "same",
                        "data_validation": {"same_source_allowed": True},
                    }
                )
            )
            torch.save(torch.tensor([0, 1, 2]), artifacts / "train_indices.pt")
            torch.save(torch.tensor([2, 3, 4]), artifacts / "reference_indices.pt")
            with self.assertRaisesRegex(ValueError, "indices overlap"):
                validate_saved_train_reference(
                    run_dir,
                    torch.zeros(3, 2),
                    torch.ones(3, 2),
                )

        config = synthetic_train_runner.build_config(
            {
                **synthetic_train_runner.default_settings(),
                "eval_sample_bank_size": 5,
                "wandb": False,
            }
        )
        with self.assertRaisesRegex(ValueError, "Saved z_init bank is smaller"):
            eval_z_init_bank(config, torch.randn(4, int(config.data.data_dim)))

    def test_same_seed_tiny_runs_reproduce_z_init_and_checkpoints(self):
        def run_once(root, eval_sample_bank_size=5):
            settings = synthetic_train_runner.default_settings()
            settings.update(
                {
                    "seed": 123,
                    "n_train": 6,
                    "n_test": 6,
                    "batch_cap": 3,
                    "eval_sample_bank_size": eval_sample_bank_size,
                    "sampling_steps": 2,
                    "ambient_dim": 4,
                    "intrinsic_data_dim": 2,
                    "n_clusters": 2,
                    "hidden_dim": 16,
                    "n_layers": 2,
                    "time_emb_dim": 16,
                    "n_training_steps": 1,
                    "ckpt_every": 1,
                    "log_every": 1,
                    "wandb": False,
                }
            )
            config = synthetic_train_runner.build_config(settings)
            with contextlib.redirect_stdout(io.StringIO()):
                return Path(synthetic_train_runner.run_full_job(config, root, "cpu")["run_dir"])

        with tempfile.TemporaryDirectory() as tmpdir:
            first = run_once(Path(tmpdir) / "a")
            second = run_once(Path(tmpdir) / "b")
            larger_bank = run_once(Path(tmpdir) / "c", eval_sample_bank_size=7)

            self.assertTrue(
                torch.equal(
                    torch.load(first / "artifacts" / "z_init.pt", map_location="cpu"),
                    torch.load(second / "artifacts" / "z_init.pt", map_location="cpu"),
                )
            )
            for checkpoint_name in ("step_000000.pt", "step_000001.pt"):
                first_state = torch.load(
                    first / "checkpoints" / checkpoint_name,
                    map_location="cpu",
                )["model"]
                second_state = torch.load(
                    second / "checkpoints" / checkpoint_name,
                    map_location="cpu",
                )["model"]
                self.assertEqual(first_state.keys(), second_state.keys())
                for key in first_state:
                    self.assertTrue(torch.equal(first_state[key], second_state[key]), key)

            first_init = torch.load(
                first / "checkpoints" / "step_000000.pt",
                map_location="cpu",
            )["model"]
            larger_bank_init = torch.load(
                larger_bank / "checkpoints" / "step_000000.pt",
                map_location="cpu",
            )["model"]
            for key in first_init:
                self.assertTrue(
                    torch.equal(first_init[key], larger_bank_init[key]),
                    f"z_init bank size changed model init: {key}",
                )

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
            (ROOT / "prototyping/runners/notebooks/synthetic_train_runner.ipynb", synthetic_sections),
            (ROOT / "prototyping/runners/notebooks/image_train_runner.ipynb", recipe_sections),
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
