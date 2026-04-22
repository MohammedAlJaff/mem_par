import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

try:
    from mp_config import get_base_config
    from mp_training import (
        bonnaire_1m_checkpoint_steps,
        parse_checkpoint_steps,
        resolve_checkpoint_steps,
        train_run,
    )
except ModuleNotFoundError:
    get_base_config = None
    bonnaire_1m_checkpoint_steps = None
    resolve_checkpoint_steps = None
    train_run = None


@unittest.skipIf(
    get_base_config is None or train_run is None,
    "project modules are required",
)
class CheckpointScheduleTests(unittest.TestCase):
    def test_bonnaire_1m_schedule_matches_expected_shape(self):
        steps = bonnaire_1m_checkpoint_steps(1_000_000)

        self.assertEqual(len(steps), 72)
        self.assertEqual(steps[0], 0)
        self.assertEqual(steps[-1], 1_000_000)
        self.assertEqual(tuple(sorted(set(steps))), steps)

    def test_bonnaire_1m_schedule_clips_to_short_runs(self):
        config = _small_config()
        config.training.n_steps = 1200
        config.training.ckpt_schedule = "bonnaire-1m"
        config.training.ckpt_steps = ()

        steps = resolve_checkpoint_steps(config)

        self.assertEqual(steps[0], 0)
        self.assertEqual(steps[-1], 1200)
        self.assertLessEqual(max(steps), 1200)

    def test_interval_schedule_includes_final_step(self):
        config = _small_config()
        config.training.n_steps = 5
        config.training.ckpt_every = 2

        self.assertEqual(resolve_checkpoint_steps(config), (0, 2, 4, 5))

    def test_explicit_checkpoint_steps_have_priority(self):
        config = _small_config()
        config.training.n_steps = 3
        config.training.ckpt_every = 2
        config.training.ckpt_schedule = "bonnaire-1m"
        config.training.ckpt_steps = (0, 1, 3)

        self.assertEqual(resolve_checkpoint_steps(config), (0, 1, 3))

    def test_explicit_checkpoint_steps_reject_bad_values(self):
        config = _small_config()
        config.training.n_steps = 3
        with self.assertRaisesRegex(ValueError, "comma-separated"):
            parse_checkpoint_steps("1,,2")

        config.training.ckpt_steps = (0, 4)
        with self.assertRaisesRegex(ValueError, "outside valid range"):
            resolve_checkpoint_steps(config)

    def test_training_saves_only_resolved_checkpoint_steps(self):
        config = _small_config()
        config.training.n_steps = 3
        config.training.ckpt_every = 2
        config.training.ckpt_steps = (0, 1, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                run_dir = Path(train_run(config, tmpdir, "cpu"))

            checkpoint_names = sorted(
                path.name for path in (run_dir / "checkpoints").glob("*.pt")
            )
            run_summary = json.loads(
                (run_dir / "artifacts" / "run_summary.json").read_text()
            )
            status = json.loads((run_dir / "status.json").read_text())

        self.assertEqual(
            checkpoint_names,
            ["step_000000.pt", "step_000001.pt", "step_000003.pt"],
        )
        self.assertEqual(
            run_summary["checkpoint_schedule"]["resolved_ckpt_steps"],
            [0, 1, 3],
        )
        self.assertEqual(
            status["checkpoint_schedule"]["resolved_ckpt_steps"],
            [0, 1, 3],
        )


def _small_config():
    """Return a tiny synthetic config for checkpoint-schedule tests."""
    config = get_base_config()
    config.study.name = "checkpoint_schedule_test"
    config.outputs.run_name = "checkpoint_schedule_test"

    config.data.dataset = "gmm_subspace"
    config.data.root = "synthetic"
    config.data.n_train = 4
    config.data.n_eval = 4
    config.data.ambient_dim = 2
    config.data.intrinsic_data_dim = 2
    config.data.data_dim = 2
    config.data.intrinsic_dim = 2
    config.data.n_clusters = 2
    config.data.sigma_z = 0.2
    config.data.sigma_eta = 0.0

    config.network.arch = "mlp"
    config.network.hidden_dim = 8
    config.network.n_layers = 1
    config.network.time_emb_dim = 8

    config.training.seed = 0
    config.training.n_steps = 3
    config.training.batch_cap = 2
    config.training.batch_size = 2
    config.training.lr = 1.0e-3
    config.training.ckpt_every = 2
    config.training.ckpt_schedule = "interval"
    config.training.ckpt_steps = ()
    config.training.log_every = 1

    config.evaluation.n_samples = 2
    config.evaluation.batch_size = 2
    config.evaluation.sample_panel_count = 2
    config.evaluation.loss_eval_batch_size = 2
    config.sampling.n_steps = 2
    config.tracking.wandb.enabled = False
    config.tracking.console.stage_banners = False
    config.tracking.console.summary_every_seconds = 1.0e9
    return config


if __name__ == "__main__":
    unittest.main()
