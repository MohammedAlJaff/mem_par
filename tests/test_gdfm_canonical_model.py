import unittest

try:
    import torch
    from ml_collections import ConfigDict
except ModuleNotFoundError:
    torch = None
    ConfigDict = None

from gdfm import GDFM


@unittest.skipIf(torch is None or ConfigDict is None, "torch and ml_collections are required")
class TestCanonicalGDFM(unittest.TestCase):
    def make_config(self):
        config = ConfigDict()

        config.framework = ConfigDict()
        config.framework.family = "fm"
        config.framework.prediction_space = "x"
        config.framework.loss_space = "x"

        config.schedule = ConfigDict()
        config.schedule.t_min = 1.0e-3
        config.schedule.t_max = 0.999
        config.schedule.denom_eps = 0.05
        config.schedule.beta_min = 0.1
        config.schedule.beta_max = 20.0
        config.schedule.time_sampling = "uniform"
        config.schedule.logit_mu = 0.0
        config.schedule.logit_sigma = 1.0

        config.network = ConfigDict()
        config.network.arch = "mlp"
        config.network.hidden_dim = 32
        config.network.n_layers = 2
        config.network.time_emb_dim = 16

        config.data = ConfigDict()
        config.data.data_dim = 8
        config.data.image_size = 0
        config.data.image_channels = 0

        return config

    def test_compute_loss_backward(self):
        model = GDFM(self.make_config())
        x = torch.randn(4, 8, requires_grad=False)
        loss = model.compute_loss(x)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        grad_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_norm += float(parameter.grad.norm().item())
        self.assertGreater(grad_norm, 0.0)

    def test_sampling_shape(self):
        model = GDFM(self.make_config())
        z = torch.randn(3, 8)
        sample = model.sample(z, n_steps=4, method="heun")
        self.assertEqual(tuple(sample.shape), (3, 8))
        self.assertTrue(torch.isfinite(sample).all().item())


if __name__ == "__main__":
    unittest.main()
