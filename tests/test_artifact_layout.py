import tempfile
import unittest
from pathlib import Path

try:
    from mp_artifacts import (
        create_run_layout,
        find_latest_attempt_dir,
        find_latest_completed_attempt_dir,
        load_json,
        mark_attempt_done,
        make_run_id,
        save_json,
        update_attempt_status,
    )
    from mp_config import clone_config, get_base_config
except ModuleNotFoundError:
    create_run_layout = None
    find_latest_attempt_dir = None
    find_latest_completed_attempt_dir = None
    load_json = None
    mark_attempt_done = None
    make_run_id = None
    save_json = None
    update_attempt_status = None
    clone_config = None
    get_base_config = None


@unittest.skipIf(
    create_run_layout is None or clone_config is None,
    "artifact helpers and config helpers are required",
)
class TestArtifactLayout(unittest.TestCase):
    def make_config(self):
        config = clone_config(get_base_config())
        config.study.name = "artifact_layout_demo"
        config.outputs.run_name = "mnist_like_run"
        config.training.seed = 7
        return config

    def test_same_run_and_seed_create_separate_attempts(self):
        config = self.make_config()
        run_id = make_run_id(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "artifact_layout_demo"
            first_attempt = create_run_layout(experiment_dir, config)
            update_attempt_status(first_attempt, "completed")

            second_attempt = create_run_layout(experiment_dir, config)
            update_attempt_status(second_attempt, "training")

            self.assertNotEqual(first_attempt, second_attempt)
            self.assertEqual(find_latest_attempt_dir(experiment_dir, run_id, 7), second_attempt)
            self.assertEqual(
                find_latest_completed_attempt_dir(experiment_dir, run_id, 7),
                first_attempt,
            )

            experiment_manifest = load_json(experiment_dir / "experiment_manifest.json")
            run_manifest = load_json(first_attempt.parents[3] / "run_manifest.json")
            seed_manifest = load_json(first_attempt.parents[1] / "seed_manifest.json")

            self.assertIn(run_id, experiment_manifest["runs"])
            self.assertEqual(seed_manifest["latest_attempt"], second_attempt.name)
            self.assertEqual(seed_manifest["latest_completed_attempt"], first_attempt.name)
            self.assertEqual(
                run_manifest["seeds"]["seed_0007"]["latest_attempt"],
                second_attempt.name,
            )
            self.assertEqual(
                run_manifest["seeds"]["seed_0007"]["latest_completed_attempt"],
                first_attempt.name,
            )

    def test_mark_attempt_done_renames_folder_and_preserves_lookup(self):
        config = self.make_config()
        run_id = make_run_id(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "artifact_layout_demo"
            attempt = create_run_layout(experiment_dir, config)
            original_name = attempt.name
            save_json(
                attempt / "artifacts" / "run_summary.json",
                {
                    "run_dir": str(attempt.resolve()),
                    "attempt_id": original_name,
                    "status_path": str((attempt / "status.json").resolve()),
                    "checkpoint_sample_previews_path": str(
                        (
                            attempt
                            / "metrics"
                            / "checkpoint_sample_previews.json"
                        ).resolve()
                    ),
                    "checkpoint_sample_previews": [
                        {
                            "step": 1,
                            "samples_path": str(
                                (
                                    attempt
                                    / "metrics"
                                    / "checkpoint_sample_previews"
                                    / "step_000001.pt"
                                ).resolve()
                            ),
                            "figure_path": str(
                                (
                                    attempt
                                    / "figures"
                                    / "checkpoint_samples_over_train_step_000001.png"
                                ).resolve()
                            ),
                        }
                    ],
                    "generation_monitor": {
                        "z_init_path": str(
                            (attempt / "artifacts" / "z_init.pt").resolve()
                        ),
                        "checkpoint_samples": [
                            {
                                "step": 1,
                                "path": str(
                                    (
                                        attempt
                                        / "figures"
                                        / "generation_monitor_samples_step_000001.png"
                                    ).resolve()
                                ),
                                "z_init_path": str(
                                    (attempt / "artifacts" / "z_init.pt").resolve()
                                ),
                            }
                        ],
                        "final_samples_path": str(
                            (
                                attempt
                                / "figures"
                                / "generation_monitor_samples_final.png"
                            ).resolve()
                        ),
                    },
                },
            )

            done_attempt = mark_attempt_done(attempt)
            self.assertEqual(done_attempt.name, f"DONE_{original_name}")
            self.assertFalse(attempt.exists())
            self.assertTrue((done_attempt / "status.json").exists())

            status = load_json(done_attempt / "status.json")
            self.assertEqual(status["attempt_id"], done_attempt.name)
            self.assertEqual(status["original_attempt_id"], original_name)
            self.assertEqual(status["attempt_dir"], str(done_attempt.resolve()))
            self.assertEqual(
                status["status_path"],
                str((done_attempt / "status.json").resolve()),
            )

            run_summary = load_json(done_attempt / "artifacts" / "run_summary.json")
            self.assertEqual(run_summary["run_dir"], str(done_attempt.resolve()))
            self.assertEqual(run_summary["attempt_id"], done_attempt.name)
            self.assertEqual(run_summary["original_attempt_id"], original_name)
            self.assertTrue(
                run_summary["generation_monitor"]["checkpoint_samples"][0][
                    "path"
                ].startswith(str(done_attempt.resolve()))
            )
            self.assertTrue(
                run_summary["checkpoint_sample_previews_path"].startswith(
                    str(done_attempt.resolve())
                )
            )
            self.assertTrue(
                run_summary["checkpoint_sample_previews"][0]["samples_path"].startswith(
                    str(done_attempt.resolve())
                )
            )
            self.assertTrue(
                run_summary["checkpoint_sample_previews"][0]["figure_path"].startswith(
                    str(done_attempt.resolve())
                )
            )

            self.assertEqual(find_latest_attempt_dir(experiment_dir, run_id, 7), done_attempt)
            update_attempt_status(done_attempt, "completed")
            self.assertEqual(
                find_latest_completed_attempt_dir(experiment_dir, run_id, 7),
                done_attempt,
            )

            seed_manifest = load_json(done_attempt.parents[1] / "seed_manifest.json")
            run_manifest = load_json(done_attempt.parents[3] / "run_manifest.json")
            self.assertIn(done_attempt.name, seed_manifest["attempts"])
            self.assertNotIn(original_name, seed_manifest["attempts"])
            self.assertEqual(seed_manifest["latest_attempt"], done_attempt.name)
            self.assertEqual(seed_manifest["latest_completed_attempt"], done_attempt.name)
            self.assertEqual(
                run_manifest["seeds"]["seed_0007"]["latest_completed_attempt"],
                done_attempt.name,
            )

    def test_failed_attempt_is_not_marked_done_implicitly(self):
        config = self.make_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "artifact_layout_demo"
            attempt = create_run_layout(experiment_dir, config)
            update_attempt_status(attempt, "failed", last_error="boom")

            self.assertTrue(attempt.exists())
            self.assertFalse(attempt.name.startswith("DONE_"))
            status = load_json(attempt / "status.json")
            self.assertEqual(status["state"], "failed")
            self.assertNotIn("original_attempt_id", status)

    def test_tracking_settings_do_not_change_run_identity(self):
        config = self.make_config()
        tracked = self.make_config()
        tracked.tracking.wandb.enabled = True
        tracked.tracking.wandb.mode = "offline"
        tracked.tracking.console.summary_every_steps = 123
        tracked.tracking.console.summary_every_seconds = 7.5

        self.assertEqual(make_run_id(config), make_run_id(tracked))


if __name__ == "__main__":
    unittest.main()
