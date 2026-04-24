import math
import unittest

try:
    import torch
    from mp_metrics.metrics_synthetic import (
        gmm_intrinsic_log_prob,
        gmm_intrinsic_metrics,
        intrinsic_sliced_w2,
        project_to_subspace,
    )
    from mp_config import clone_config, get_base_config
    from mp_data import build_data_bundle
except ModuleNotFoundError:
    torch = None
    gmm_intrinsic_log_prob = None
    gmm_intrinsic_metrics = None
    intrinsic_sliced_w2 = None
    project_to_subspace = None
    clone_config = None
    get_base_config = None
    build_data_bundle = None


@unittest.skipIf(
    torch is None or build_data_bundle is None,
    "torch and the projected GMM data stack are required",
)
class TestProjectedGMMIntrinsicMetrics(unittest.TestCase):
    def make_config(self, sigma_eta=1.0e-3):
        config = clone_config(get_base_config())
        config.data.dataset = "gmm_subspace"
        config.data.n_train = 10
        config.data.n_test = 12
        config.data.data_dim = 8
        config.data.intrinsic_dim = 2
        config.data.n_clusters = 3
        config.data.sigma_z = 0.1
        config.data.sigma_eta = sigma_eta
        config.data.geometry_seed = 5
        config.training.seed = 7
        return config

    def test_manual_equations_match_canonical_loader(self):
        config = self.make_config()
        canonical = build_data_bundle(config)

        geometry_generator = torch.Generator().manual_seed(int(config.data.geometry_seed))
        basis, _ = torch.linalg.qr(
            torch.randn(
                int(config.data.data_dim),
                int(config.data.intrinsic_dim),
                generator=geometry_generator,
            )
        )
        basis = basis[:, : int(config.data.intrinsic_dim)]
        means = torch.randn(
            int(config.data.n_clusters),
            int(config.data.intrinsic_dim),
            generator=geometry_generator,
        )

        train_generator = torch.Generator().manual_seed(int(config.training.seed))
        train_labels = torch.randint(
            int(config.data.n_clusters),
            (int(config.data.n_train),),
            generator=train_generator,
        )
        train_latent = means[train_labels] + float(config.data.sigma_z) * torch.randn(
            int(config.data.n_train),
            int(config.data.intrinsic_dim),
            generator=train_generator,
        )
        train = train_latent @ basis.T
        train = train + float(config.data.sigma_eta) * torch.randn(
            int(config.data.n_train),
            int(config.data.data_dim),
            generator=train_generator,
        )

        reference_generator = torch.Generator().manual_seed(int(config.training.seed) + 1)
        reference_labels = torch.randint(
            int(config.data.n_clusters),
            (int(config.data.n_test),),
            generator=reference_generator,
        )
        reference_latent = means[reference_labels] + float(config.data.sigma_z) * torch.randn(
            int(config.data.n_test),
            int(config.data.intrinsic_dim),
            generator=reference_generator,
        )
        reference = reference_latent @ basis.T
        reference = reference + float(config.data.sigma_eta) * torch.randn(
            int(config.data.n_test),
            int(config.data.data_dim),
            generator=reference_generator,
        )

        self.assertTrue(torch.equal(train.contiguous(), canonical["train"]))
        self.assertTrue(torch.equal(reference.contiguous(), canonical["reference"]))
        self.assertTrue(torch.equal(train_labels, canonical["train_labels"]))
        self.assertTrue(torch.equal(reference_labels, canonical["reference_labels"]))
        self.assertTrue(torch.equal(basis, canonical["geometry"]["basis"]))
        self.assertTrue(torch.equal(means, canonical["geometry"]["means"]))

    def test_projection_shapes_for_subspace_case(self):
        bundle = build_data_bundle(self.make_config())
        train = bundle["train"]
        basis = bundle["geometry"]["basis"]
        parts = project_to_subspace(train, basis)

        self.assertEqual(tuple(parts["intrinsic"].shape), (10, 2))
        self.assertEqual(tuple(parts["projected"].shape), (10, 8))

    def test_intrinsic_gmm_nll_is_finite_when_ambient_noise_is_positive(self):
        config = self.make_config(sigma_eta=1.0e-3)
        bundle = build_data_bundle(config)
        samples = bundle["reference"]
        basis = bundle["geometry"]["basis"]
        means = bundle["geometry"]["means"]
        intrinsic = project_to_subspace(samples, basis)["intrinsic"]
        directions = torch.eye(int(config.data.intrinsic_dim))

        metrics = gmm_intrinsic_metrics(
            samples,
            basis,
            means,
            float(config.data.sigma_z),
            float(config.data.sigma_eta),
            intrinsic,
            directions,
        )

        self.assertTrue(math.isfinite(metrics["intrinsic_gmm_nll_mean"]))
        self.assertTrue(math.isfinite(metrics["intrinsic_sliced_w2"]))

    def test_intrinsic_gmm_nll_is_finite_for_singular_subspace(self):
        config = self.make_config(sigma_eta=0.0)
        bundle = build_data_bundle(config)
        samples = bundle["reference"]
        basis = bundle["geometry"]["basis"]
        means = bundle["geometry"]["means"]
        intrinsic = project_to_subspace(samples, basis)["intrinsic"]
        directions = torch.eye(int(config.data.intrinsic_dim))

        metrics = gmm_intrinsic_metrics(
            samples,
            basis,
            means,
            float(config.data.sigma_z),
            float(config.data.sigma_eta),
            intrinsic,
            directions,
        )

        self.assertTrue(math.isfinite(metrics["intrinsic_gmm_nll_mean"]))
        self.assertNotIn("oracle_full_nll_mean", metrics)

    def test_intrinsic_nll_prefers_points_near_true_components(self):
        means = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
        near = torch.tensor([[0.05, -0.05], [4.05, 0.03]])
        far = torch.tensor([[12.0, 12.0], [-10.0, 9.0]])

        near_log_prob = gmm_intrinsic_log_prob(near, means, variance=0.1)
        far_log_prob = gmm_intrinsic_log_prob(far, means, variance=0.1)

        self.assertLess(float((-near_log_prob).mean().item()), float((-far_log_prob).mean().item()))

    def test_intrinsic_sliced_w2_is_zero_for_exact_matches(self):
        means = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        directions = torch.eye(2)

        self.assertAlmostEqual(intrinsic_sliced_w2(means, means, directions), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
