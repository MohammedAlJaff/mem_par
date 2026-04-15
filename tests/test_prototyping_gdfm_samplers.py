"""
Sidecar sampler tests for the current GDFM prototype.

The goal here is not to replace the existing in-model sampler yet. Instead we
prove that an external sampler surface can already wrap the current model
correctly, so future sampler refactors have a working reference.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

from tests.gdfm_sampler_contract import (
    assert_external_sampler_matches_builtin,
    assert_vp_probability_flow_matches_velocity,
    assert_vp_reverse_sde_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPING_DIR = ROOT / "prototyping"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch is not installed in this environment.")
class TestPrototypingGDFMSamplers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gdfm = load_module(
            "gdfm_components_sampler_test",
            PROTOTYPING_DIR / "gdfm_components.py",
        )
        cls.samplers = load_module(
            "gdfm_samplers_sampler_test",
            PROTOTYPING_DIR / "gdfm_samplers.py",
        )

    def make_config(self, **overrides):
        config = dict(
            family="fm",
            prediction_space="v",
            loss_space="v",
            data_dim=2,
            network="mlp",
            hidden_dim=128,
            n_layers=3,
            time_emb_dim=32,
            t_min=1e-3,
            t_max=0.999,
            safe_div_eps=0.05,
            fm_time_sampling="uniform",
            fm_logit_mu=0.0,
            fm_logit_sigma=1.0,
            beta_min=0.1,
            beta_max=20.0,
        )
        config.update(overrides)
        return SimpleNamespace(**config)

    def make_model(self, **overrides):
        torch.manual_seed(0)
        return self.gdfm.GDFM(self.make_config(**overrides))

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
                    self.samplers,
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
                    self.samplers,
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
            self.samplers,
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
            self.samplers,
            batch_size=32,
            n_steps=15,
        )
        self.assertEqual(out.shape, (32, model.config.data_dim))


if __name__ == "__main__":
    unittest.main()
