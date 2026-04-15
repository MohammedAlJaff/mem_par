"""
Sidecar sampler tests for the canonical GDFM package.

Keep the working samplers in `gdfm.core` for now. These tests only check that
the external sampler module in `gdfm.samplers` can wrap the canonical model
with the same behavior as the in-model reference.
"""

import unittest

try:
    import torch
    from ml_collections import ConfigDict
    import gdfm.samplers as samplers
    from gdfm import GDFM
except ModuleNotFoundError:
    torch = None
    ConfigDict = None
    samplers = None
    GDFM = None
from tests.gdfm_sampler_contract import (
    assert_external_sampler_matches_builtin,
    assert_vp_probability_flow_matches_velocity,
    assert_vp_reverse_sde_contract,
)


@unittest.skipIf(
    torch is None or ConfigDict is None or samplers is None or GDFM is None,
    "torch, ml_collections, and gdfm are required",
)
class TestCanonicalGDFMSamplers(unittest.TestCase):
    def make_config(self, **overrides):
        config = ConfigDict()

        config.framework = ConfigDict()
        config.framework.family = "fm"
        config.framework.prediction_space = "v"
        config.framework.loss_space = "v"

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
        config.network.hidden_dim = 128
        config.network.n_layers = 3
        config.network.time_emb_dim = 32

        config.data = ConfigDict()
        config.data.data_dim = 8
        config.data.image_size = 0
        config.data.image_channels = 0

        for key, value in overrides.items():
            if hasattr(config.framework, key):
                config.framework[key] = value
            elif hasattr(config.schedule, key):
                config.schedule[key] = value
            elif hasattr(config.network, key):
                config.network[key] = value
            elif hasattr(config.data, key):
                config.data[key] = value
            else:
                config[key] = value

        return config

    def make_model(self, **overrides):
        torch.manual_seed(0)
        return GDFM(self.make_config(**overrides))

    def test_external_euler_matches_builtin(self):
        for family in ("fm", "vp"):
            with self.subTest(family=family):
                model = self.make_model(
                    family=family,
                    prediction_space="x",
                    loss_space="x",
                )
                err = assert_external_sampler_matches_builtin(
                    model,
                    samplers,
                    method="euler",
                    atol=1e-6,
                )
                self.assertLess(err, 1e-6)

    def test_external_heun_matches_builtin(self):
        for family in ("fm", "vp"):
            with self.subTest(family=family):
                model = self.make_model(
                    family=family,
                    prediction_space="eps",
                    loss_space="eps",
                )
                err = assert_external_sampler_matches_builtin(
                    model,
                    samplers,
                    method="heun",
                    atol=1e-6,
                )
                self.assertLess(err, 1e-6)

    def test_vp_probability_flow_bridge_matches_velocity(self):
        model = self.make_model(
            family="vp",
            prediction_space="v",
            loss_space="v",
        )
        err = assert_vp_probability_flow_matches_velocity(
            model,
            samplers,
            atol=5e-5,
        )
        self.assertLess(err, 5e-5)

    def test_vp_reverse_sde_shape_and_finiteness(self):
        model = self.make_model(
            family="vp",
            prediction_space="eps",
            loss_space="eps",
        )
        out = assert_vp_reverse_sde_contract(
            model,
            samplers,
            batch_size=32,
            n_steps=15,
        )
        self.assertEqual(tuple(out.shape), (32, model.config.data.data_dim))


if __name__ == "__main__":
    unittest.main()
