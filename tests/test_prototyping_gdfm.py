"""
Contract tests for the current prototyping GDFM implementation.

These tests are inspired by the sanity-check notebook and are intended to be
reused for future GDFM variants. New models can either adapt this file or call
the shared helpers in tests/gdfm_contract.py directly.
"""

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

from tests.gdfm_contract import (
    assert_conversion_cycles,
    assert_loss_backward,
    assert_prediction_shape,
    assert_sampling_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPING_DIR = ROOT / "prototyping"
PROTOTYPE_GDFM_PATH = PROTOTYPING_DIR / "gdfm_components.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch is not installed in this environment.")
class TestPrototypingGDFM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PROTOTYPE_GDFM_PATH.is_file():
            raise unittest.SkipTest(
                "active prototyping/gdfm_components.py is not present in this layout"
            )
        cls.gdfm = load_module(
            "gdfm_components_under_test",
            PROTOTYPE_GDFM_PATH,
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

    def test_prediction_shape_preserved(self):
        for family in ("fm", "vp"):
            with self.subTest(family=family):
                model = self.make_model(family=family)
                assert_prediction_shape(model)

    def test_full_loss_backward(self):
        for family in ("fm", "vp"):
            with self.subTest(family=family):
                model = self.make_model(family=family, prediction_space="x", loss_space="v")
                assert_loss_backward(model)

    def test_conversion_cycle_checks(self):
        for family in ("fm", "vp"):
            with self.subTest(family=family):
                model = self.make_model(
                    family=family,
                    prediction_space="v",
                    loss_space="v",
                    fm_time_sampling="logit_normal" if family == "fm" else "uniform",
                )
                worst_err = assert_conversion_cycles(model, atol=1e-4)
                self.assertLess(worst_err, 1e-4)

    def test_sampling_shape_and_finiteness(self):
        for family in ("fm", "vp"):
            with self.subTest(family=family):
                model = self.make_model(family=family, prediction_space="x", loss_space="x")
                outputs = assert_sampling_contract(model, batch_size=32, n_steps=15)
                self.assertEqual(outputs["euler"].shape, outputs["heun"].shape)

    def test_mlp_is_loaded_from_gdfm_package(self):
        model = self.make_model()
        self.assertEqual(model.net.__class__.__module__, "gdfm.networks.mlp")
        self.assertEqual(self.gdfm.build_network.__module__, "gdfm.model_setup")

    def test_debug_helpers_smoke(self):
        model = self.make_model()
        x = torch.randn(8, model.config.data_dim)
        t = model.sample_t(8, x.device)

        with contextlib.redirect_stdout(io.StringIO()):
            self.gdfm.print_model_summary(model)
            self.gdfm.print_forward_shapes(model, x, t)


if __name__ == "__main__":
    unittest.main()
