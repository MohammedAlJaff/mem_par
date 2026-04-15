"""
Model setup helpers for GDFM-family models.

This module owns the decision of which backbone class to instantiate from a
config-like object. The point is to make model setup testable on its own,
independently of the surrounding GDFM model.

It accepts both:
    - flat configs with fields like cfg.network and cfg.data_dim
    - nested configs with cfg.network.arch and cfg.data.image_size

The canonical public shape is:
    cfg.network.arch
    cfg.network.*
    cfg.data.*

During transition we also accept legacy aliases so experiment-shaped configs can
be adapted into this factory without rewriting the rest of the repo first.
"""

from collections.abc import Mapping

from gdfm.networks import BonnaireUNet, JiT, TimeCondMLP

_MISSING = object()


def _get_value(container, name, default=_MISSING):
    if container is _MISSING or container is None:
        return default

    if isinstance(container, Mapping):
        return container.get(name, default)

    return getattr(container, name, default)


def _named_section(cfg, *names):
    for name in names:
        value = _get_value(cfg, name, _MISSING)
        if value is not _MISSING:
            return value
    return _MISSING


def _search_value(sources, aliases, label, default=_MISSING):
    for source in sources:
        for alias in aliases:
            value = _get_value(source, alias, _MISSING)
            if value is not _MISSING:
                return value

    if default is not _MISSING:
        return default

    raise ValueError(f"Config is missing required field: {label}")


def _tuple_or_default(value, default=()):
    if value is _MISSING or value is None:
        return tuple(default)
    return tuple(value)


def _network_name(cfg):
    network_cfg = _named_section(cfg, "network")
    if isinstance(network_cfg, str):
        return network_cfg

    arch = _get_value(network_cfg, "arch", _MISSING)
    if arch is not _MISSING:
        return arch

    model_cfg = _named_section(cfg, "model")
    arch = _get_value(model_cfg, "arch", _MISSING)
    if arch is not _MISSING:
        return arch

    arch = _get_value(cfg, "arch", _MISSING)
    if arch is not _MISSING:
        return arch

    raise ValueError(
        "Config must define network.arch or a flat network string."
    )


def _normalize_network_config(cfg):
    arch = _network_name(cfg)
    network_cfg = _named_section(cfg, "network")
    model_cfg = _named_section(cfg, "model")
    data_cfg = _named_section(cfg, "data")

    network_sources = [network_cfg, model_cfg, cfg]
    data_sources = [data_cfg, model_cfg, network_cfg, cfg]

    spec = {"arch": arch}

    if arch == "mlp":
        spec["data_dim"] = _search_value(
            data_sources,
            ("data_dim", "D"),
            "data.data_dim",
        )
        spec["hidden_dim"] = _search_value(
            network_sources,
            ("hidden_dim",),
            "network.hidden_dim",
        )
        spec["n_layers"] = _search_value(
            network_sources,
            ("n_layers", "depth"),
            "network.n_layers",
        )
        spec["time_emb_dim"] = _search_value(
            network_sources,
            ("time_emb_dim", "time_embed_dim"),
            "network.time_emb_dim",
            default=64,
        )
        return spec

    if arch == "unet":
        spec["image_size"] = _search_value(
            data_sources,
            ("image_size", "resolution"),
            "data.image_size",
        )
        spec["image_channels"] = _search_value(
            data_sources,
            ("image_channels", "in_channels"),
            "data.image_channels",
        )
        spec["base_channels"] = _search_value(
            network_sources,
            ("base_channels",),
            "network.base_channels",
        )
        spec["channel_mult"] = _tuple_or_default(
            _search_value(
                network_sources,
                ("channel_mult", "channel_mults"),
                "network.channel_mult",
                default=(1, 2, 4, 4),
            )
        )
        spec["attn_resolutions"] = _tuple_or_default(
            _search_value(
                network_sources,
                ("attn_resolutions", "attention_resolutions"),
                "network.attn_resolutions",
                default=(),
            )
        )
        spec["num_res_blocks"] = _search_value(
            network_sources,
            ("num_res_blocks",),
            "network.num_res_blocks",
            default=2,
        )
        spec["dropout"] = _search_value(
            network_sources,
            ("dropout",),
            "network.dropout",
            default=0.0,
        )
        spec["time_emb_dim"] = _search_value(
            network_sources,
            ("time_emb_dim", "time_embed_dim"),
            "network.time_emb_dim",
            default=None,
        )
        return spec

    if arch == "jit":
        spec["image_size"] = _search_value(
            data_sources,
            ("image_size", "resolution"),
            "data.image_size",
        )
        spec["image_channels"] = _search_value(
            data_sources,
            ("image_channels", "in_channels"),
            "data.image_channels",
        )
        spec["patch_size"] = _search_value(
            network_sources,
            ("patch_size",),
            "network.patch_size",
        )
        spec["hidden_dim"] = _search_value(
            network_sources,
            ("hidden_dim",),
            "network.hidden_dim",
        )
        spec["depth"] = _search_value(
            network_sources,
            ("depth",),
            "network.depth",
        )
        spec["heads"] = _search_value(
            network_sources,
            ("heads", "num_heads"),
            "network.heads",
        )
        spec["bottleneck_dim"] = _search_value(
            network_sources,
            ("bottleneck_dim",),
            "network.bottleneck_dim",
        )
        spec["proj_dropout"] = _search_value(
            network_sources,
            ("proj_dropout", "dropout"),
            "network.proj_dropout",
            default=0.0,
        )
        spec["attn_dropout"] = _search_value(
            network_sources,
            ("attn_dropout", "attn_drop"),
            "network.attn_dropout",
            default=0.0,
        )
        spec["in_context_len"] = _search_value(
            network_sources,
            ("in_context_len",),
            "network.in_context_len",
            default=32,
        )
        spec["in_context_start"] = _search_value(
            network_sources,
            ("in_context_start",),
            "network.in_context_start",
            default=4,
        )
        return spec

    raise ValueError(f"Unknown network: {arch}")


def build_network(cfg):
    """
    Instantiate the backbone selected by cfg.network.

    Canonical config shape:
        cfg.network.arch = "mlp" | "unet" | "jit"
        cfg.network.*
        cfg.data.*

    Compatibility inputs:
        - flat configs
        - dict-like configs
        - legacy aliases such as model.*, channel_mults, attention_resolutions,
          num_heads, in_channels, resolution, D, depth, and time_embed_dim
    """
    spec = _normalize_network_config(cfg)
    network_name = spec["arch"]

    if network_name == "mlp":
        return TimeCondMLP(
            data_dim=spec["data_dim"],
            hidden_dim=spec["hidden_dim"],
            n_layers=spec["n_layers"],
            time_emb_dim=spec["time_emb_dim"],
        )

    if network_name == "unet":
        return BonnaireUNet(
            image_size=spec["image_size"],
            image_channels=spec["image_channels"],
            base_channels=spec["base_channels"],
            channel_mult=spec["channel_mult"],
            num_res_blocks=spec["num_res_blocks"],
            attn_resolutions=spec["attn_resolutions"],
            dropout=spec["dropout"],
            time_emb_dim=spec["time_emb_dim"],
        )

    if network_name == "jit":
        return JiT(
            image_size=spec["image_size"],
            image_channels=spec["image_channels"],
            patch_size=spec["patch_size"],
            hidden_dim=spec["hidden_dim"],
            depth=spec["depth"],
            heads=spec["heads"],
            bottleneck_dim=spec["bottleneck_dim"],
            proj_dropout=spec["proj_dropout"],
            attn_dropout=spec["attn_dropout"],
            in_context_len=spec["in_context_len"],
            in_context_start=spec["in_context_start"],
        )

    raise ValueError(f"Unknown network: {network_name}")


__all__ = ["build_network", "BonnaireUNet", "JiT", "TimeCondMLP"]
