import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from mp_config import clone_config, get_base_config
    from mp_metrics.checkpoint_previews import (
        checkpoint_preview_figure_name,
        collect_checkpoint_sample_preview_record,
    )
except ModuleNotFoundError:
    torch = None
    clone_config = None
    get_base_config = None
    checkpoint_preview_figure_name = None
    collect_checkpoint_sample_preview_record = None


@unittest.skipIf(
    torch is None or collect_checkpoint_sample_preview_record is None,
    "torch and checkpoint preview helpers are required",
)
class TestCheckpointPreviews(unittest.TestCase):
    def make_config(self):
        config = clone_config(get_base_config())
        config.evaluation.sample_panel_count = 3
        config.visual_sanity.enabled = False
        return config

    def test_saves_tensor_preview_for_vector_samples(self):
        config = self.make_config()
        samples = torch.arange(20, dtype=torch.float32).view(10, 2)
        train = torch.zeros(10, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            record = collect_checkpoint_sample_preview_record(
                step=12,
                samples=samples,
                train_tensor=train,
                run_dir=tmpdir,
                config=config,
            )

            self.assertEqual(record["step"], 12)
            self.assertEqual(record["count"], 3)
            self.assertIsNone(record["figure_path"])
            saved = torch.load(record["samples_path"], map_location="cpu")
            self.assertTrue(torch.equal(saved, samples[:3]))

    def test_saves_image_panel_when_visual_sanity_is_enabled(self):
        config = self.make_config()
        config.visual_sanity.enabled = True
        config.visual_sanity.num_images = 2
        samples = torch.rand(4, 1, 8, 8)
        train = torch.rand(4, 1, 8, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            record = collect_checkpoint_sample_preview_record(
                step=5,
                samples=samples,
                train_tensor=train,
                run_dir=tmpdir,
                config=config,
            )

            self.assertEqual(record["count"], 2)
            self.assertTrue(Path(record["samples_path"]).is_file())
            self.assertTrue(Path(record["figure_path"]).is_file())
            expected_name = checkpoint_preview_figure_name(5)
            self.assertTrue(record["figure_path"].endswith(f"{expected_name}.png"))

    def test_skips_image_panel_for_non_image_tensors(self):
        config = self.make_config()
        config.visual_sanity.enabled = True
        samples = torch.randn(4, 3)
        train = torch.randn(4, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            record = collect_checkpoint_sample_preview_record(
                step=1,
                samples=samples,
                train_tensor=train,
                run_dir=tmpdir,
                config=config,
            )

            self.assertTrue(Path(record["samples_path"]).is_file())
            self.assertIsNone(record["figure_path"])


if __name__ == "__main__":
    unittest.main()
