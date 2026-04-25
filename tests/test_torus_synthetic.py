import contextlib
import math
import io
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from mp_data import build_data_bundle
    from mp_data.torus import (
        torus_angles_to_xyz,
        torus_periodic_features,
    )
    from mp_runners import eval_synthetic, synthetic_train_runner
    from mp_metrics.metrics_synthetic import (
        collect_controlled_synthetic_metric_record,
        intrinsic_geometry_parts,
        torus3d_metrics,
    )
except ModuleNotFoundError:
    torch = None
    build_data_bundle = None
    torus_angles_to_xyz = None
    torus_periodic_features = None
    synthetic_train_runner = None
    eval_synthetic = None
    collect_controlled_synthetic_metric_record = None
    intrinsic_geometry_parts = None
    torus3d_metrics = None


@unittest.skipIf(
    torch is None or build_data_bundle is None,
    "torch and the synthetic torus stack are required",
)
class TestTorusSynthetic(unittest.TestCase):
    def make_config(self, sigma_eta=0.0):
        settings = synthetic_train_runner.default_settings()
        settings.update(
            {
                "dataset": "torus3d",
                "ambient_dim": 3,
                "n_train": 16,
                "n_test": 18,
                "eval_sample_bank_size": 8,
                "sigma_eta": sigma_eta,
                "major_radius": 2.0,
                "minor_radius": 0.5,
                "sliced_w2_projections": 4,
            }
        )
        return synthetic_train_runner.build_config(settings)

    def test_torus3d_bundle_contains_angles_and_geometry(self):
        bundle = build_data_bundle(self.make_config())

        self.assertEqual(tuple(bundle["train"].shape), (16, 3))
        self.assertEqual(tuple(bundle["reference"].shape), (18, 3))
        self.assertEqual(str(bundle["geometry"]["kind"]), "torus3d")
        self.assertEqual(tuple(bundle["geometry"]["basis"].shape), (3, 3))
        self.assertEqual(tuple(bundle["geometry"]["train_torus_xyz"].shape), (16, 3))
        self.assertEqual(int(bundle["geometry"]["local_manifold_dim"]), 2)
        self.assertEqual(int(bundle["geometry"]["metric_feature_dim"]), 4)
        self.assertEqual(tuple(bundle["geometry"]["train_angles"].shape), (16, 2))
        self.assertEqual(tuple(bundle["geometry"]["train_intrinsic"].shape), (16, 4))

    def test_torus3d_can_embed_canonical_donut_into_larger_ambient_dimension(self):
        config = self.make_config(sigma_eta=0.0)
        config.data.ambient_dim = 16
        config.data.data_dim = 16
        config.data.intrinsic_data_dim = 16
        config.data.intrinsic_dim = 16
        bundle = build_data_bundle(config)
        geometry = bundle["geometry"]

        self.assertEqual(tuple(bundle["train"].shape), (16, 16))
        self.assertEqual(tuple(bundle["reference"].shape), (18, 16))
        self.assertEqual(tuple(geometry["basis"].shape), (16, 3))
        recovered = bundle["train"] @ geometry["basis"]

        self.assertTrue(torch.allclose(recovered, geometry["train_torus_xyz"], atol=1.0e-6))

    def test_periodic_intrinsic_representation_wraps_angles(self):
        theta = torch.tensor([0.0, 2.0 * math.pi])
        phi = torch.tensor([0.0, 2.0 * math.pi])

        features = torus_periodic_features(theta, phi)

        self.assertTrue(torch.allclose(features[0], features[1], atol=1.0e-6))

    def test_torus_points_have_near_zero_geometry_error(self):
        config = self.make_config(sigma_eta=0.0)
        bundle = build_data_bundle(config)
        geometry = bundle["geometry"]
        angles = geometry["reference_angles"]
        torus_xyz = torus_angles_to_xyz(
            angles[:, 0],
            angles[:, 1],
            float(geometry["major_radius"]),
            float(geometry["minor_radius"]),
        )
        samples = torus_xyz @ geometry["basis"].T
        parts = intrinsic_geometry_parts(samples, geometry, config=config)

        self.assertLess(float(parts["geometry_error"].mean().item()), 1.0e-6)

    def test_torus_metrics_include_intrinsic_ratios_and_sliced_w2(self):
        config = self.make_config(sigma_eta=0.0)
        bundle = build_data_bundle(config)
        geometry = bundle["geometry"]
        train_intrinsic = intrinsic_geometry_parts(
            bundle["train"],
            geometry,
            config=config,
        )["intrinsic"]
        reference_parts = intrinsic_geometry_parts(
            bundle["reference"],
            geometry,
            config=config,
        )
        directions = torch.eye(4)

        torus_metrics = torus3d_metrics(
            bundle["reference"],
            geometry,
            reference_parts["intrinsic"],
            directions,
        )
        record = collect_controlled_synthetic_metric_record(
            step=0,
            samples=bundle["reference"][:8],
            baseline_samples=bundle["reference"][:8],
            train_tensor=bundle["train"],
            train_intrinsic=train_intrinsic,
            geometry=geometry,
            reference_intrinsic=reference_parts["intrinsic"][:8],
            sliced_directions=directions,
            chunk_size=64,
            config=config,
        )

        self.assertIn("intrinsic_sliced_w2", torus_metrics)
        self.assertIn("intrinsic_nearest_neighbor_ratio_d1_over_d2", record)
        self.assertIn("holdout_intrinsic_nearest_neighbor_ratio_d1_over_d2", record)
        self.assertIn("intrinsic_sliced_w2", record)
        self.assertNotIn("angle_coverage_l1", record)
        self.assertNotIn("torus_distance_mean", record)
        self.assertNotIn("geometry_error_mean", record)

    def test_torus_eval_replay_writes_train_and_checkpoint_donut_figures(self):
        settings = synthetic_train_runner.default_settings()
        settings.update(
            {
                "dataset": "torus3d",
                "ambient_dim": 3,
                "n_train": 8,
                "n_test": 8,
                "batch_cap": 4,
                "eval_sample_bank_size": 4,
                "sampling_steps": 2,
                "hidden_dim": 16,
                "n_layers": 2,
                "time_emb_dim": 16,
                "sliced_w2_projections": 4,
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
                result = eval_synthetic.run_eval_job(train_result["run_dir"], "cpu")

            figures = result["figures"]
            self.assertTrue(
                Path(figures["torus3d_train_set_canonical_3d_scatter"]).is_file()
            )
            self.assertNotIn(
                "torus3d_canonical_scatter_train_reference_generated_3d",
                figures,
            )
            self.assertNotIn(
                "torus3d_xy_projection_scatter_train_reference_generated",
                figures,
            )
            self.assertNotIn("intrinsic_distance_vs_step_checkpoints", figures)
            self.assertNotIn(
                "nearest_neighbor_ratio_d1_over_d2_by_checkpoint_step",
                figures,
            )
            previews = result["checkpoint_sample_previews"]
            self.assertGreater(len(previews), 0)
            self.assertTrue(Path(previews[-1]["figure_path"]).is_file())
            self.assertIn(
                "torus3d_checkpoint_train_faded_generated",
                previews[-1]["figure_path"],
            )


if __name__ == "__main__":
    unittest.main()
