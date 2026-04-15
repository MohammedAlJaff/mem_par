"""
Tests for the canonical GDFM-side network setup entrypoint.

These stay at the network-factory layer. The question is simple:
if a config asks for MLP, UNet, or JiT, does gdfm.build_network return the
right backbone with the right forward contract?
"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

from gdfm import build_network


@unittest.skipIf(torch is None, "torch is not installed in this environment.")
class TestGDFMModelSetup(unittest.TestCase):
    def make_canonical_config(self, arch="mlp", **overrides):
        network = SimpleNamespace(
            arch=arch,
            hidden_dim=64,
            n_layers=3,
            time_emb_dim=32,
            base_channels=32,
            channel_mult=(1, 2, 4, 4),
            attn_resolutions=(8,),
            num_res_blocks=2,
            dropout=0.0,
            patch_size=4,
            depth=2,
            heads=4,
            bottleneck_dim=32,
            proj_dropout=0.0,
            attn_dropout=0.0,
            in_context_len=32,
            in_context_start=4,
        )
        data = SimpleNamespace(
            data_dim=8,
            image_size=16,
            image_channels=1,
        )
        cfg = SimpleNamespace(network=network, data=data)

        for key, value in overrides.items():
            if hasattr(network, key):
                setattr(network, key, value)
            elif hasattr(data, key):
                setattr(data, key, value)
            else:
                setattr(cfg, key, value)

        return cfg

    def make_flat_mlp_config(self, **overrides):
        cfg = dict(
            network="mlp",
            data_dim=11,
            hidden_dim=64,
            n_layers=3,
            time_emb_dim=32,
        )
        cfg.update(overrides)
        return SimpleNamespace(**cfg)

    def make_gmm_like_config(self):
        return {
            "network": "mlp",
            "model": {
                "hidden_dim": 256,
                "depth": 3,
                "time_embed_dim": 64,
            },
            "data": {
                "D": 256,
            },
        }

    def make_celeba_like_config(self):
        return {
            "network": "unet",
            "model": {
                "in_channels": 1,
                "base_channels": 64,
                "channel_mults": [1, 2, 4, 4],
                "attention_resolutions": [8],
                "num_res_blocks": 2,
                "dropout": 0.0,
            },
            "data": {
                "resolution": 32,
            },
        }

    def make_afhq_like_config(self, resolution):
        return {
            "network": "jit",
            "model": {
                "in_channels": 3,
                "patch_size": 16,
                "hidden_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "bottleneck_dim": 128,
                "dropout": 0.0,
            },
            "data": {
                "resolution": resolution,
            },
        }

    def test_canonical_nested_selects_mlp(self):
        net = build_network(self.make_canonical_config(arch="mlp"))
        self.assertEqual(net.__class__.__module__, "gdfm.networks.mlp")

    def test_canonical_nested_selects_unet(self):
        net = build_network(self.make_canonical_config(arch="unet"))
        self.assertEqual(net.__class__.__module__, "gdfm.networks.unet")
        self.assertEqual(net.model.__class__.__module__, "gdfm.networks.small_unet")

    def test_canonical_nested_selects_jit(self):
        net = build_network(
            self.make_canonical_config(
                arch="jit",
                image_size=16,
                image_channels=3,
                patch_size=4,
                hidden_dim=64,
                depth=2,
                heads=4,
                bottleneck_dim=32,
            )
        )
        self.assertEqual(net.__class__.__module__, "gdfm.networks.jit")

    def test_flat_mlp_compatibility_config_builds(self):
        net = build_network(self.make_flat_mlp_config())
        z = torch.randn(5, 11)
        t = torch.rand(5)
        out = net(z, t)
        self.assertEqual(tuple(out.shape), (5, 11))

    def test_synthetic_gmm_like_mlp_builds_and_runs(self):
        net = build_network(self.make_gmm_like_config())
        z = torch.randn(2, 256)
        t = torch.rand(2)
        with torch.no_grad():
            out = net(z, t)
        self.assertEqual(tuple(out.shape), (2, 256))

    def test_celeba_like_unet_builds_and_runs(self):
        net = build_network(self.make_celeba_like_config())
        z = torch.randn(2, 1, 32, 32)
        t = torch.rand(2)
        with torch.no_grad():
            out = net(z, t)
        self.assertEqual(tuple(out.shape), (2, 1, 32, 32))

    def test_afhq64_like_jit_builds_and_runs(self):
        net = build_network(self.make_afhq_like_config(64))
        z = torch.randn(1, 3, 64, 64)
        t = torch.rand(1)
        with torch.no_grad():
            out = net(z, t)
        self.assertEqual(tuple(out.shape), (1, 3, 64, 64))

    def test_afhq128_like_jit_builds_and_runs(self):
        net = build_network(self.make_afhq_like_config(128))
        z = torch.randn(1, 3, 128, 128)
        t = torch.rand(1)
        with torch.no_grad():
            out = net(z, t)
        self.assertEqual(tuple(out.shape), (1, 3, 128, 128))

    def test_afhq256_like_jit_builds(self):
        net = build_network(self.make_afhq_like_config(256))
        self.assertEqual(net.image_size, 256)
        self.assertEqual(net.patch_size, 16)
        self.assertEqual(net.hidden_dim, 768)

    def test_unknown_network_fails_clearly(self):
        cfg = self.make_canonical_config(arch="mlp")
        cfg.network.arch = "mystery"
        with self.assertRaises(ValueError):
            build_network(cfg)

    def test_missing_required_unet_fields_fail_clearly(self):
        cfg = self.make_celeba_like_config()
        del cfg["model"]["in_channels"]
        with self.assertRaises(ValueError):
            build_network(cfg)

    def test_jit_patch_mismatch_fails_clearly(self):
        cfg = self.make_afhq_like_config(64)
        cfg["model"]["patch_size"] = 18
        with self.assertRaises(ValueError):
            build_network(cfg)

    def test_mlp_forward_batch_mismatch_fails_clearly(self):
        net = build_network(self.make_flat_mlp_config())
        with self.assertRaises(ValueError):
            net(torch.randn(4, 11), torch.rand(5))

    def test_unet_forward_rank_check_fails_clearly(self):
        net = build_network(self.make_celeba_like_config())
        with self.assertRaises(ValueError):
            net(torch.randn(2, 1, 32), torch.rand(2))

    def test_jit_forward_batch_mismatch_fails_clearly(self):
        net = build_network(
            self.make_canonical_config(
                arch="jit",
                image_size=16,
                image_channels=3,
                patch_size=4,
                hidden_dim=64,
                depth=2,
                heads=4,
                bottleneck_dim=32,
            )
        )
        with self.assertRaises(ValueError):
            net(torch.randn(2, 3, 16, 16), torch.rand(3))

    def test_missing_vendored_jit_path_fails_clearly(self):
        cfg = self.make_afhq_like_config(64)
        with mock.patch("gdfm.networks.jit.VENDOR_ROOT", Path("/definitely/missing")):
            with self.assertRaises(FileNotFoundError):
                build_network(cfg)


if __name__ == "__main__":
    unittest.main()
