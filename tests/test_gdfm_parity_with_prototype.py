import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from gdfm import GDFM as CanonicalGDFM
except ModuleNotFoundError:
    torch = None
    CanonicalGDFM = None


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPING_DIR = ROOT / "prototyping"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(
    torch is None or CanonicalGDFM is None,
    "torch and the canonical gdfm package are required",
)
class TestGDFMParityWithPrototype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prototype_module = load_module(
            "gdfm_components_parity_test",
            PROTOTYPING_DIR / "gdfm_components.py",
        )

    def make_config(self, **overrides):
        config = dict(
            family="fm",
            prediction_space="v",
            loss_space="v",
            data_dim=8,
            network="mlp",
            hidden_dim=64,
            n_layers=3,
            time_emb_dim=16,
            t_min=1.0e-3,
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

    def make_pair(self, **overrides):
        config = self.make_config(**overrides)
        prototype = self.prototype_module.GDFM(config)
        canonical = CanonicalGDFM(config)
        canonical.load_state_dict(prototype.state_dict(), strict=True)
        prototype.eval()
        canonical.eval()
        return prototype, canonical

    def assert_close(self, left, right, atol=1.0e-6, rtol=1.0e-6):
        self.assertTrue(
            torch.allclose(left, right, atol=atol, rtol=rtol),
            msg=f"max abs diff = {(left - right).abs().max().item()}",
        )

    def test_state_dict_is_compatible(self):
        prototype, canonical = self.make_pair()
        self.assertEqual(
            sorted(prototype.state_dict().keys()),
            sorted(canonical.state_dict().keys()),
        )

    def test_compute_loss_matches_prototype(self):
        for family in ("fm", "vp"):
            with self.subTest(family=family):
                prototype, canonical = self.make_pair(
                    family=family,
                    prediction_space="x",
                    loss_space="v",
                )
                x = torch.randn(5, 8)
                eps = torch.randn_like(x)
                t = torch.linspace(0.1, 0.9, 5)
                loss_prototype = prototype.compute_loss(x, eps=eps, t=t)
                loss_canonical = canonical.compute_loss(x, eps=eps, t=t)
                self.assert_close(loss_prototype, loss_canonical)

    def test_convert_matches_prototype(self):
        for family in ("fm", "vp"):
            with self.subTest(family=family):
                prototype, canonical = self.make_pair(family=family)
                x = torch.randn(4, 8)
                eps = torch.randn_like(x)
                t = torch.tensor([0.15, 0.35, 0.55, 0.75])
                z_t = prototype.z_t_interpolate(x, eps, t)

                source_tensors = {
                    "x": torch.randn_like(x),
                    "eps": torch.randn_like(x),
                    "v": torch.randn_like(x),
                }

                for source in ("x", "eps", "v"):
                    for target in ("x", "eps", "v"):
                        with self.subTest(family=family, source=source, target=target):
                            converted_prototype = prototype.convert(
                                source_tensors[source], z_t, t, source, target
                            )
                            converted_canonical = canonical.convert(
                                source_tensors[source], z_t, t, source, target
                            )
                            self.assert_close(converted_prototype, converted_canonical)

    def test_sampling_matches_prototype(self):
        for family in ("fm", "vp"):
            with self.subTest(family=family):
                prototype, canonical = self.make_pair(
                    family=family,
                    prediction_space="x",
                    loss_space="x",
                )
                z = torch.randn(3, 8)
                for method in ("euler", "heun"):
                    with self.subTest(family=family, method=method):
                        sample_prototype = prototype.sample(z.clone(), n_steps=6, method=method)
                        sample_canonical = canonical.sample(z.clone(), n_steps=6, method=method)
                        self.assert_close(sample_prototype, sample_canonical, atol=1.0e-5, rtol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
