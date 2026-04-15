import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from mp_config import get_base_config
    from mp_metrics import (
        compute_f_mem,
        compute_folder_metrics,
        compute_memorization_metrics,
        compute_nn_stats,
    )
except ModuleNotFoundError:
    torch = None
    get_base_config = None
    compute_f_mem = None
    compute_folder_metrics = None
    compute_memorization_metrics = None
    compute_nn_stats = None


@unittest.skipIf(
    torch is None or get_base_config is None or compute_memorization_metrics is None,
    "torch and the memorization metric stack are required",
)
class TestMPMetrics(unittest.TestCase):
    def test_threshold_can_come_from_config(self):
        reference = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]],
            dtype=torch.float32,
        )
        samples = torch.tensor(
            [[0.05, 0.05], [1.5, 1.5]],
            dtype=torch.float32,
        )
        stats = compute_nn_stats(samples, reference)

        config = get_base_config()
        config.evaluation.ratio_threshold = 0.8

        f_mem_loose = compute_f_mem(stats["r"], config=config)
        f_mem_strict = compute_f_mem(stats["r"], threshold=0.2)

        self.assertGreaterEqual(f_mem_loose, f_mem_strict)

    def test_direct_tensor_metrics_report_threshold(self):
        reference = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]],
            dtype=torch.float32,
        )
        copied = reference[:2].clone()

        metrics = compute_memorization_metrics(
            copied,
            reference,
            ratio_threshold=0.5,
        )

        self.assertEqual(metrics["ratio_threshold"], 0.5)
        self.assertEqual(metrics["n_samples"], 2)
        self.assertEqual(metrics["n_reference"], 3)
        self.assertGreaterEqual(metrics["f_mem"], 0.5)

    def test_folder_metrics_match_tensor_metrics_for_pt_batches(self):
        reference = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [5.0, 5.0]],
            dtype=torch.float32,
        )
        samples = torch.tensor(
            [[0.0, 0.0], [1.1, 1.1], [7.0, 7.0]],
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            samples_dir = tmpdir / "samples"
            reference_dir = tmpdir / "reference"
            samples_dir.mkdir()
            reference_dir.mkdir()

            torch.save(samples[:2], samples_dir / "batch_a.pt")
            torch.save(samples[2:].clone(), samples_dir / "batch_b.pt")
            torch.save(reference[:2], reference_dir / "ref_a.pt")
            torch.save(reference[2:], reference_dir / "ref_b.pt")

            direct = compute_memorization_metrics(
                samples,
                reference,
                ratio_threshold=0.5,
            )
            folder = compute_folder_metrics(
                samples_dir,
                reference_dir,
                ratio_threshold=0.5,
            )

        self.assertAlmostEqual(folder["f_mem"], direct["f_mem"], places=6)
        self.assertAlmostEqual(folder["mean_nn_ratio"], direct["mean_nn_ratio"], places=6)
        self.assertAlmostEqual(folder["mean_d1"], direct["mean_d1"], places=6)
        self.assertAlmostEqual(folder["mean_d2"], direct["mean_d2"], places=6)


if __name__ == "__main__":
    unittest.main()
