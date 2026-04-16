import tempfile
import unittest
from pathlib import Path

try:
    from mp_artifacts import (
        create_run_layout,
        find_latest_attempt_dir,
        find_latest_completed_attempt_dir,
        load_json,
        make_run_id,
        update_attempt_status,
    )
    from mp_config import clone_config, get_base_config
except ModuleNotFoundError:
    create_run_layout = None
    find_latest_attempt_dir = None
    find_latest_completed_attempt_dir = None
    load_json = None
    make_run_id = None
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


if __name__ == "__main__":
    unittest.main()
