import tempfile
import unittest
from unittest import mock
from pathlib import Path

try:
    import torch
    from PIL import Image
    from mp_config import get_base_config
    from mp_metrics.images import (
        compute_f_mem_by_rho,
        compute_f_mem,
        compute_image_fid,
        compute_image_memorization_metrics,
        compute_memorization_metrics,
        compute_nn_stats,
        collect_image32_metric_record,
        RHO_SWEEP,
    )
    from mp_metrics.images import _to_uint8_rgb
    from mp_runners.metric_io import compute_image_folder_memorization_metrics
except ModuleNotFoundError:
    torch = None
    Image = None
    get_base_config = None
    compute_f_mem_by_rho = None
    compute_f_mem = None
    compute_image_fid = None
    compute_image_folder_memorization_metrics = None
    compute_image_memorization_metrics = None
    compute_memorization_metrics = None
    compute_nn_stats = None
    collect_image32_metric_record = None
    RHO_SWEEP = None
    _to_uint8_rgb = None


@unittest.skipIf(
    torch is None or get_base_config is None or compute_memorization_metrics is None,
    "torch and the memorization metric stack are required",
)
class TestMPMetrics(unittest.TestCase):
    def test_nn_stats_match_bruteforce_top2(self):
        reference = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0]],
            dtype=torch.float32,
        )
        samples = torch.tensor(
            [[0.1, 0.0], [2.8, 0.0], [6.5, 0.0]],
            dtype=torch.float32,
        )

        distances = torch.cdist(samples, reference)
        expected = torch.topk(distances, k=2, largest=False, dim=1)

        stats = compute_nn_stats(
            samples,
            reference,
            query_batch_size=1,
            reference_batch_size=2,
        )

        self.assertEqual(set(stats.keys()), {"d1", "d2", "r", "nn_idx"})
        self.assertTrue(torch.allclose(stats["d1"], expected.values[:, 0]))
        self.assertTrue(torch.allclose(stats["d2"], expected.values[:, 1]))
        self.assertTrue(torch.equal(stats["nn_idx"], expected.indices[:, 0]))
        self.assertTrue(torch.allclose(stats["r"], stats["d1"] / stats["d2"]))

    def test_nn_stats_are_chunk_invariant(self):
        torch.manual_seed(7)
        samples = torch.randn(9, 5)
        reference = torch.randn(13, 5)

        full = compute_nn_stats(
            samples,
            reference,
            query_batch_size=9,
        )
        chunked = compute_nn_stats(
            samples,
            reference,
            query_batch_size=2,
            reference_batch_size=3,
        )

        for key in ("d1", "d2", "r"):
            self.assertTrue(torch.allclose(chunked[key], full[key], atol=1.0e-6))
        self.assertTrue(torch.equal(chunked["nn_idx"], full["nn_idx"]))

    def test_rho_can_come_from_config(self):
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
        f_mem_strict = compute_f_mem(stats["r"], rho=0.2)

        self.assertGreaterEqual(f_mem_loose, f_mem_strict)

    def test_direct_tensor_metrics_report_rho(self):
        reference = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]],
            dtype=torch.float32,
        )
        copied = reference[:2].clone()

        metrics = compute_memorization_metrics(
            copied,
            reference,
            rho=0.5,
        )

        self.assertEqual(
            set(metrics.keys()),
            {
                "rho",
                "n_samples",
                "N",
                "f_mem",
                "mean_r",
                "median_r",
                "mean_d1",
                "mean_d2",
                "median_d1",
                "median_d2",
            },
        )
        self.assertEqual(metrics["rho"], 0.5)
        self.assertEqual(metrics["n_samples"], 2)
        self.assertEqual(metrics["N"], 3)
        self.assertGreaterEqual(metrics["f_mem"], 0.5)
        self.assertIn("median_r", metrics)

    def test_image_checkpoint_record_uses_canonical_keys_and_delta(self):
        config = get_base_config()
        config.evaluation.ratio_threshold = 0.5
        config.evaluation.chunk_size = 8

        train = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.3, 0.3], [0.3, 0.3]]],
                [[[0.8, 0.8], [0.8, 0.8]]],
            ],
            dtype=torch.float32,
        )
        samples = train[:2].clone()
        baseline = train[[1, 2]].clone()

        record = collect_image32_metric_record(
            5,
            samples,
            baseline,
            train,
            chunk_size=8,
            config=config,
        )

        self.assertEqual(record["step"], 5)
        self.assertIn("rho", record)
        self.assertIn("mean_r", record)
        self.assertIn("median_r", record)
        self.assertIn("delta_f_mem", record)
        self.assertNotIn("ratio_threshold", record)
        self.assertNotIn("mean_nn_ratio", record)
        self.assertAlmostEqual(
            record["delta_f_mem"],
            record["f_mem"] - record["baseline_f_mem"],
            places=6,
        )
        for rho in RHO_SWEEP:
            suffix = f"rho{rho}"
            self.assertIn(f"f_mem_{suffix}", record)
            self.assertIn(f"baseline_f_mem_{suffix}", record)
            self.assertIn(f"delta_f_mem_{suffix}", record)

    def test_folder_metrics_match_tensor_metrics_for_pt_batches(self):
        reference = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.25, 0.25], [0.25, 0.25]]],
                [[[0.75, 0.75], [0.75, 0.75]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
        samples = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.30, 0.30], [0.30, 0.30]]],
                [[[0.90, 0.90], [0.90, 0.90]]],
            ],
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

            direct = compute_image_memorization_metrics(
                samples,
                reference,
                rho=0.5,
                image_range="0_1",
            )
            folder = compute_image_folder_memorization_metrics(
                samples_dir,
                reference_dir,
                rho=0.5,
                image_range="0_1",
            )

        self.assertAlmostEqual(folder["f_mem"], direct["f_mem"], places=6)
        self.assertAlmostEqual(folder["mean_r"], direct["mean_r"], places=6)
        self.assertAlmostEqual(folder["median_r"], direct["median_r"], places=6)
        self.assertAlmostEqual(folder["mean_d1"], direct["mean_d1"], places=6)
        self.assertAlmostEqual(folder["mean_d2"], direct["mean_d2"], places=6)

    def test_image_folder_metrics_match_tensor_metrics_for_pngs(self):
        reference = torch.tensor(
            [
                [[[0, 0], [0, 0]]],
                [[[32, 32], [32, 32]]],
                [[[96, 96], [96, 96]]],
                [[[192, 192], [192, 192]]],
            ],
            dtype=torch.uint8,
        )
        samples = torch.tensor(
            [
                [[[0, 0], [0, 0]]],
                [[[40, 40], [40, 40]]],
                [[[200, 200], [200, 200]]],
            ],
            dtype=torch.uint8,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            samples_dir = tmpdir / "samples"
            reference_dir = tmpdir / "reference"
            samples_dir.mkdir()
            reference_dir.mkdir()
            for index, image in enumerate(samples):
                Image.fromarray(image.squeeze(0).numpy()).save(samples_dir / f"{index:03d}.png")
            for index, image in enumerate(reference):
                Image.fromarray(image.squeeze(0).numpy()).save(reference_dir / f"{index:03d}.png")

            direct = compute_image_memorization_metrics(
                samples,
                reference,
                rho=0.5,
                query_batch_size=1,
                reference_batch_size=2,
                image_range="0_255",
            )
            folder = compute_image_folder_memorization_metrics(
                samples_dir,
                reference_dir,
                rho=0.5,
                query_batch_size=1,
                reference_batch_size=2,
                image_range="0_255",
            )

        self.assertAlmostEqual(folder["f_mem"], direct["f_mem"], places=6)
        self.assertAlmostEqual(folder["mean_r"], direct["mean_r"], places=6)
        self.assertEqual(folder["space"], "pixels")
        self.assertEqual(folder["feature_dim"], 4)

    def test_image_feature_metrics_accept_dummy_encoder(self):
        class DummyEncoder(torch.nn.Module):
            def forward(self, batch):
                return batch.flatten(1)[:, :2] * 2.0

        reference = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.25, 0.25], [0.25, 0.25]]],
                [[[0.75, 0.75], [0.75, 0.75]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ]
        )
        samples = reference[[0, 2]].clone()
        encoder = DummyEncoder()

        image_metrics = compute_image_memorization_metrics(
            samples,
            reference,
            encoder=encoder,
            rho=0.5,
            query_batch_size=1,
            reference_batch_size=2,
            image_range="0_1",
        )
        direct_metrics = compute_memorization_metrics(
            encoder(samples).detach(),
            encoder(reference).detach(),
            rho=0.5,
            query_batch_size=1,
            reference_batch_size=2,
        )

        self.assertEqual(image_metrics["space"], "features")
        self.assertEqual(image_metrics["encoder"], "DummyEncoder")
        self.assertEqual(image_metrics["feature_dim"], 2)
        self.assertAlmostEqual(image_metrics["f_mem"], direct_metrics["f_mem"], places=6)
        self.assertAlmostEqual(
            image_metrics["mean_r"],
            direct_metrics["mean_r"],
            places=6,
        )

    def test_f_mem_rho_sweep(self):
        sweep = compute_f_mem_by_rho(torch.tensor([0.2, 0.6, 0.9]), [0.5, 0.8])

        self.assertAlmostEqual(sweep[0.5], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(sweep[0.8], 2.0 / 3.0, places=6)

    def test_uint8_rgb_adapter_handles_repo_normalized_grayscale(self):
        images = torch.tensor([[[[-1.0, 0.0], [1.0, 0.5]]]])

        converted = _to_uint8_rgb(images, image_range="-1_1")

        self.assertEqual(tuple(converted.shape), (1, 3, 2, 2))
        self.assertEqual(converted.dtype, torch.uint8)
        self.assertTrue(torch.equal(converted[:, 0], converted[:, 1]))
        self.assertTrue(torch.equal(converted[:, 1], converted[:, 2]))
        self.assertEqual(int(converted[0, 0, 0, 0].item()), 0)
        self.assertEqual(int(converted[0, 0, 1, 0].item()), 255)

    def test_auto_image_range_warns(self):
        images = torch.rand(1, 1, 2, 2)

        with self.assertWarnsRegex(UserWarning, "image_range='auto' inferred range"):
            _to_uint8_rgb(images, image_range="auto")

    def test_image_fid_uses_torch_fidelity_dataset_adapter(self):
        try:
            import torch_fidelity  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-fidelity is required for the FID wrapper")

        samples = torch.rand(2, 1, 4, 4)
        reference = torch.rand(3, 1, 4, 4)

        with mock.patch(
            "torch_fidelity.calculate_metrics",
            return_value={"frechet_inception_distance": 12.5},
        ) as calculate_metrics:
            metrics = compute_image_fid(
                samples,
                reference,
                batch_size=2,
                device="cpu",
                image_range="0_1",
            )

        kwargs = calculate_metrics.call_args.kwargs
        self.assertEqual(metrics["fid"], 12.5)
        self.assertEqual(metrics["frechet_inception_distance"], 12.5)
        self.assertEqual(metrics["n_samples"], 2)
        self.assertEqual(metrics["n_reference"], 3)
        self.assertFalse(kwargs["cuda"])
        self.assertFalse(kwargs.get("isc", False))
        self.assertEqual(len(kwargs["input1"]), 2)
        self.assertEqual(len(kwargs["input2"]), 3)


if __name__ == "__main__":
    unittest.main()
