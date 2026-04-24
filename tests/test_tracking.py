import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    from mp_artifacts import create_run_layout, load_json, make_run_id
    from mp_config import clone_config, get_base_config
    from mp_tracking import AttemptTracker, resolve_wandb_identity
except ModuleNotFoundError:
    create_run_layout = None
    load_json = None
    make_run_id = None
    clone_config = None
    get_base_config = None
    AttemptTracker = None
    resolve_wandb_identity = None


class FakeWandbRun:
    def __init__(self, run_id, name):
        self.id = run_id
        self.name = name
        self.path = ["entity", "project", run_id]
        self.url = f"https://wandb.local/{run_id}"
        self.logs = []
        self.finished = False

    def log(self, payload, step=None):
        self.logs.append((payload, step))

    def finish(self):
        self.finished = True


@unittest.skipIf(
    create_run_layout is None or AttemptTracker is None,
    "artifact and tracking helpers are required",
)
class TestAttemptTracker(unittest.TestCase):
    def make_config(self):
        config = clone_config(get_base_config())
        config.study.name = "tracking_demo"
        config.outputs.run_name = "tracking_demo"
        config.training.seed = 3
        config.tracking.console.enabled = False
        return config

    def install_fake_wandb(self, fake_module):
        old_module = sys.modules.get("wandb")
        sys.modules["wandb"] = fake_module
        return old_module

    def restore_wandb(self, old_module):
        if old_module is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = old_module

    def test_wandb_offline_logs_metrics_and_figures(self):
        config = self.make_config()
        config.tracking.wandb.enabled = True
        config.tracking.wandb.mode = "offline"
        fake = types.SimpleNamespace(init_calls=[], runs=[])

        def init(**kwargs):
            fake.init_calls.append(kwargs)
            run = FakeWandbRun(kwargs["id"], kwargs["name"])
            fake.runs.append(run)
            return run

        fake.init = init
        fake.Image = lambda path, caption=None: {
            "image_path": path,
            "caption": caption,
        }
        old_module = self.install_fake_wandb(fake)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                experiment_dir = Path(tmpdir) / "tracking_demo"
                attempt_dir = create_run_layout(experiment_dir, config)
                figure_path = attempt_dir / "figures" / "demo.png"
                figure_path.parent.mkdir(parents=True, exist_ok=True)
                figure_path.write_bytes(b"fake image")

                tracker = AttemptTracker(config, attempt_dir)
                tracker.log_metrics({"train/loss": 1.25}, step=4)
                tracker.register_figure("demo", figure_path)
                tracker.log_figure_metric(
                    "eval/checkpoint_samples_over_train",
                    figure_path,
                    step=7,
                )
                tracker.log_figure_metric(
                    "generation_monitor/samples",
                    figure_path,
                    step=9,
                )
                tracker.close()

                self.assertEqual(fake.init_calls[0]["id"], attempt_dir.name)
                self.assertEqual(fake.init_calls[0]["project"], "memparam-gmm_subspace-mlp")
                self.assertEqual(
                    fake.init_calls[0]["name"],
                    "gmm_subspace-mlp-N256-B64-x_x-steps200-s3",
                )
                self.assertEqual(fake.init_calls[0]["group"], "gmm_subspace-mlp-steps200")
                self.assertEqual(fake.init_calls[0]["mode"], "offline")
                self.assertTrue(fake.init_calls[0]["dir"].endswith("logs/wandb"))
                self.assertEqual(
                    fake.runs[0].logs[0],
                    ({"train/loss": 1.25, "train/step": 4}, None),
                )
                self.assertIn("figures/demo", fake.runs[0].logs[1][0])
                self.assertEqual(
                    fake.runs[0].logs[2],
                    (
                        {
                            "eval/checkpoint_samples_over_train": {
                                "image_path": str(figure_path.resolve()),
                                "caption": "checkpoint_samples_over_train | grad_step=7",
                            },
                            "eval/step": 7,
                        },
                        None,
                    ),
                )
                self.assertEqual(
                    fake.runs[0].logs[3],
                    (
                        {
                            "generation_monitor/samples": {
                                "image_path": str(figure_path.resolve()),
                                "caption": "generation_monitor | grad_step=9",
                            },
                            "generation_monitor/step": 9,
                        },
                        None,
                    ),
                )
                self.assertTrue(fake.runs[0].finished)

                status = load_json(attempt_dir / "status.json")
                self.assertTrue(status["wandb"]["enabled"])
                self.assertFalse(status["wandb"]["active"])
                self.assertEqual(status["wandb"]["run_id"], attempt_dir.name)
                self.assertEqual(
                    status["wandb"]["run_name"],
                    "gmm_subspace-mlp-N256-B64-x_x-steps200-s3",
                )
                self.assertEqual(status["wandb"]["group"], "gmm_subspace-mlp-steps200")
                self.assertEqual(status["generated_figures"]["demo"], str(figure_path.resolve()))

                replay_tracker = AttemptTracker(config, attempt_dir)
                replay_tracker.close()
                self.assertEqual(fake.init_calls[-1]["id"], attempt_dir.name)
        finally:
            self.restore_wandb(old_module)

    def test_wandb_auto_identity_and_explicit_overrides(self):
        config = self.make_config()
        config.tracking.wandb.enabled = True
        config.tracking.wandb.mode = "offline"
        fake = types.SimpleNamespace(init_calls=[], runs=[])

        def init(**kwargs):
            fake.init_calls.append(kwargs)
            run = FakeWandbRun(kwargs["id"], kwargs["name"])
            fake.runs.append(run)
            return run

        fake.init = init
        fake.Image = lambda path, caption=None: {
            "image_path": path,
            "caption": caption,
        }
        old_module = self.install_fake_wandb(fake)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                experiment_dir = Path(tmpdir) / "tracking_demo"
                with mock.patch.dict("os.environ", {"WANDB_PROJECT": "env_project"}):
                    attempt_dir = create_run_layout(experiment_dir, config)
                    tracker = AttemptTracker(config, attempt_dir)
                    tracker.close()
                self.assertEqual(fake.init_calls[-1]["project"], "memparam-gmm_subspace-mlp")
                self.assertEqual(
                    fake.init_calls[-1]["name"],
                    "gmm_subspace-mlp-N256-B64-x_x-steps200-s3",
                )
                self.assertEqual(fake.init_calls[-1]["group"], "gmm_subspace-mlp-steps200")

                explicit = self.make_config()
                explicit.tracking.wandb.enabled = True
                explicit.tracking.wandb.mode = "offline"
                explicit.tracking.wandb.project = "manual_project"
                explicit.tracking.wandb.name = "manual_name"
                explicit.tracking.wandb.group = "manual_group"
                with mock.patch.dict("os.environ", {"WANDB_PROJECT": "env_project"}):
                    attempt_dir = create_run_layout(experiment_dir, explicit)
                    tracker = AttemptTracker(explicit, attempt_dir)
                    tracker.close()
                self.assertEqual(fake.init_calls[-1]["project"], "manual_project")
                self.assertEqual(fake.init_calls[-1]["name"], "manual_name")
                self.assertEqual(fake.init_calls[-1]["group"], "manual_group")
        finally:
            self.restore_wandb(old_module)

    def test_wandb_auto_identity_recomputes_when_config_changes(self):
        config = self.make_config()
        config.tracking.wandb.project_explicit = False
        config.tracking.wandb.name_explicit = False
        config.tracking.wandb.group_explicit = False

        resolve_wandb_identity(config)
        self.assertIn("-x_x-", config.tracking.wandb.name)

        config.framework.prediction_space = "eps"
        config.framework.loss_space = "eps"
        identity = resolve_wandb_identity(config)

        self.assertEqual(
            identity["name"],
            "gmm_subspace-mlp-N256-B64-eps_eps-steps200-s3",
        )
        self.assertEqual(config.tracking.wandb.name, identity["name"])

    def test_wandb_explicit_marker_keeps_manual_name(self):
        config = self.make_config()
        config.tracking.wandb.project = "manual_project"
        config.tracking.wandb.name = "manual_name"
        config.tracking.wandb.group = "manual_group"
        config.tracking.wandb.project_explicit = True
        config.tracking.wandb.name_explicit = True
        config.tracking.wandb.group_explicit = True

        config.framework.prediction_space = "eps"
        config.framework.loss_space = "eps"
        identity = resolve_wandb_identity(config)

        self.assertEqual(identity["project"], "manual_project")
        self.assertEqual(identity["name"], "manual_name")
        self.assertEqual(identity["group"], "manual_group")

    def test_wandb_init_failure_is_nonfatal_by_default(self):
        config = self.make_config()
        config.tracking.wandb.enabled = True
        config.tracking.wandb.strict = False

        def init(**kwargs):
            raise RuntimeError("offline store unavailable")

        fake = types.SimpleNamespace(init=init, Image=lambda path: path)
        old_module = self.install_fake_wandb(fake)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                experiment_dir = Path(tmpdir) / "tracking_demo"
                attempt_dir = create_run_layout(experiment_dir, config)
                tracker = AttemptTracker(config, attempt_dir)
                tracker.close()

                status = load_json(attempt_dir / "status.json")
                self.assertTrue(status["wandb"]["enabled"])
                self.assertFalse(status["wandb"]["active"])
                self.assertIn("W&B initialization failed", status["wandb"]["warning"])
                self.assertTrue(status["warnings"])
                self.assertIn("warning: W&B initialization failed", (attempt_dir / "logs" / "attempt.log").read_text())
        finally:
            self.restore_wandb(old_module)

    def test_interactive_progress_keeps_summaries_out_of_terminal(self):
        config = self.make_config()
        config.tracking.console.enabled = True
        config.tracking.wandb.enabled = False

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "tracking_demo"
            attempt_dir = create_run_layout(experiment_dir, config)
            tracker = AttemptTracker(config, attempt_dir)
            tracker.interactive_progress = True

            with mock.patch("builtins.print") as printed:
                tracker.training_progress(
                    step=1,
                    total_steps=4,
                    loss=1.25,
                    elapsed_sec=0.5,
                    force=True,
                )
                tracker.checkpoint_saved(
                    step=1,
                    checkpoint_path=attempt_dir / "checkpoints" / "step_000001.pt",
                )

            printed.assert_not_called()
            attempt_log = (attempt_dir / "logs" / "attempt.log").read_text()
            self.assertIn("train step=1/4", attempt_log)
            self.assertIn("saved checkpoint step=1", attempt_log)


if __name__ == "__main__":
    unittest.main()
