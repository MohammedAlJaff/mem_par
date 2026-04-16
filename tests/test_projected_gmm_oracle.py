import math
import unittest

try:
    import torch
    from mp_config import clone_config, get_base_config
    from mp_data import build_data_bundle
except ModuleNotFoundError:
    torch = None
    clone_config = None
    get_base_config = None
    build_data_bundle = None


@unittest.skipIf(
    torch is None or build_data_bundle is None,
    "torch and the projected GMM data stack are required",
)
class TestProjectedGMMOracle(unittest.TestCase):
    def make_config(self, sigma_eta=1.0e-3):
        config = clone_config(get_base_config())
        config.data.dataset = "gmm_subspace"
        config.data.n_train = 10
        config.data.n_eval = 12
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
            (int(config.data.n_eval),),
            generator=reference_generator,
        )
        reference_latent = means[reference_labels] + float(config.data.sigma_z) * torch.randn(
            int(config.data.n_eval),
            int(config.data.intrinsic_dim),
            generator=reference_generator,
        )
        reference = reference_latent @ basis.T
        reference = reference + float(config.data.sigma_eta) * torch.randn(
            int(config.data.n_eval),
            int(config.data.data_dim),
            generator=reference_generator,
        )

        self.assertTrue(torch.equal(train.contiguous(), canonical["train"]))
        self.assertTrue(torch.equal(reference.contiguous(), canonical["reference"]))
        self.assertTrue(torch.equal(train_labels, canonical["train_labels"]))
        self.assertTrue(torch.equal(reference_labels, canonical["reference_labels"]))
        self.assertTrue(torch.equal(basis, canonical["geometry"]["basis"]))
        self.assertTrue(torch.equal(means, canonical["geometry"]["means"]))

    def test_projection_and_residual_shapes_for_subspace_case(self):
        bundle = build_data_bundle(self.make_config())
        train = bundle["train"]
        basis = bundle["geometry"]["basis"]
        latent = train @ basis
        projected = latent @ basis.T
        residual = train - projected

        self.assertEqual(tuple(latent.shape), (10, 2))
        self.assertEqual(tuple(projected.shape), (10, 8))
        self.assertEqual(tuple(residual.shape), (10, 8))

    def test_full_oracle_nll_is_finite_when_ambient_noise_is_positive(self):
        config = self.make_config(sigma_eta=1.0e-3)
        bundle = build_data_bundle(config)
        samples = bundle["reference"]
        basis = bundle["geometry"]["basis"]
        means = bundle["geometry"]["means"]
        latent = samples @ basis
        residual = samples - latent @ basis.T

        latent_var = float(config.data.sigma_z) ** 2 + float(config.data.sigma_eta) ** 2
        latent_delta = latent[:, None, :] - means[None, :, :]
        latent_sq = latent_delta.pow(2).sum(dim=2)
        log_components = (
            -0.5
            * (
                latent_sq / latent_var
                + int(config.data.intrinsic_dim) * math.log(2.0 * math.pi * latent_var)
            )
            - math.log(float(config.data.n_clusters))
        )
        latent_nll = -torch.logsumexp(log_components, dim=1)
        residual_dim = int(config.data.data_dim - config.data.intrinsic_dim)
        residual_nll = 0.5 * (
            residual.pow(2).sum(dim=1) / (float(config.data.sigma_eta) ** 2)
            + residual_dim * math.log(2.0 * math.pi * float(config.data.sigma_eta) ** 2)
        )
        full_nll = latent_nll + residual_nll

        self.assertTrue(torch.isfinite(full_nll).all())
        self.assertEqual(tuple(full_nll.shape), (12,))

    def test_full_oracle_nll_is_undefined_for_singular_subspace(self):
        config = self.make_config(sigma_eta=0.0)
        bundle = build_data_bundle(config)
        samples = bundle["reference"]
        basis = bundle["geometry"]["basis"]
        means = bundle["geometry"]["means"]
        latent = samples @ basis

        latent_var = float(config.data.sigma_z) ** 2
        latent_delta = latent[:, None, :] - means[None, :, :]
        latent_sq = latent_delta.pow(2).sum(dim=2)
        log_components = (
            -0.5
            * (
                latent_sq / latent_var
                + int(config.data.intrinsic_dim) * math.log(2.0 * math.pi * latent_var)
            )
            - math.log(float(config.data.n_clusters))
        )
        latent_nll = -torch.logsumexp(log_components, dim=1)
        full_nll = None

        self.assertTrue(torch.isfinite(latent_nll).all())
        self.assertIsNone(full_nll)


if __name__ == "__main__":
    unittest.main()
