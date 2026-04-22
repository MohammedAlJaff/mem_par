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
    """
    Read one field from either a mapping-like config or an attribute object.

    This mirrors the helper in `gdfm.core` because model setup has to accept
    the same mix of `ConfigDict`, plain dict, and attribute-style configs.
    """
    if container is _MISSING or container is None:
        return default

    if isinstance(container, Mapping):
        return container.get(name, default)

    return getattr(container, name, default)


def _named_section(cfg, *names):
    """
    Return the first config subsection that exists among `names`.
    """
    for name in names:
        value = _get_value(cfg, name, _MISSING)
        if value is not _MISSING:
            return value
    return _MISSING


def _search_value(sources, aliases, label, default=_MISSING):
    """
    Search several config sources for the first matching alias.

    `sources` is ordered from most canonical to most legacy. That ordering
    lets us gradually migrate old experiment configs without breaking the
    public network factory.
    """
    for source in sources:
        for alias in aliases:
            value = _get_value(source, alias, _MISSING)
            if value is not _MISSING:
                return value

    if default is not _MISSING:
        return default

    raise ValueError(f"Config is missing required field: {label}")


def _tuple_or_default(value, default=()):
    """
    Normalize sequence-like config fields to tuples.

    We use tuples for architectural fields such as channel multipliers and
    attention resolutions so the normalized spec is stable and easy to print.
    """
    if value is _MISSING or value is None:
        return tuple(default)
    return tuple(value)


def _network_name(cfg):
    """
    Resolve which backbone the config is asking for.

    Canonically this should come from `cfg.network.arch`, but older shapes may
    still encode it as a flat string or under `cfg.model.arch`.
    """
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
    """
    Convert mixed config layouts into one canonical backbone spec.

    The returned `spec` is the internal contract consumed by
    `build_network(...)`. Alias handling stays here so the actual constructor
    logic can stay direct and readable.
    """
    arch = _network_name(cfg)
    network_cfg = _named_section(cfg, "network")
    model_cfg = _named_section(cfg, "model")
    data_cfg = _named_section(cfg, "data")

    # We search architecture-specific fields in two groups:
    #   1. network-like sources for backbone hyperparameters
    #   2. data-like sources for image size, channels, or ambient dimension
    network_sources = [network_cfg, model_cfg, cfg]
    data_sources = [data_cfg, model_cfg, network_cfg, cfg]

    spec = {"arch": arch}

    if arch == "mlp":
        # The MLP path is the simplest: it needs one ambient dimension plus a
        # few feed-forward hyperparameters.
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
        # The UNet path normalizes the paper-facing image backbone surface. We
        # keep the constructor explicit even though the wrapped legacy module
        # still uses older argument names internally.
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
        # JiT has the richest config surface, so alias normalization matters
        # most here. In particular, we accept several older names from vendor
        # or experiment-shaped configs and map them into one stable spec.
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

    The factory is intentionally explicit: each supported architecture has a
    visible constructor call and a small validated argument set.
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


if __name__ == "__main__":
    from ml_collections import ConfigDict
    import torch
    from gdfm.core import GDFM, print_forward_shapes, print_model_summary

    torch.manual_seed(0)

    def make_config(arch):
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
        config.network.arch = arch
        config.network.hidden_dim = 32
        config.network.n_layers = 2
        config.network.time_emb_dim = 32
        config.network.base_channels = 32
        config.network.channel_mult = (1, 2, 4, 4)
        config.network.attn_resolutions = (8,)
        config.network.num_res_blocks = 2
        config.network.dropout = 0.0
        config.network.patch_size = 4
        config.network.depth = 2
        config.network.heads = 4
        config.network.bottleneck_dim = 32
        config.network.proj_dropout = 0.0
        config.network.attn_dropout = 0.0
        config.network.in_context_len = 8
        config.network.in_context_start = 1

        config.data = ConfigDict()
        config.data.data_dim = 8
        config.data.image_size = 32
        config.data.image_channels = 3
        return config

    print("build_network example")

    # Check all three factory targets under the same tiny nested config shape.
    for arch in ("mlp", "unet", "jit"):
        cfg = make_config(arch)
        model = GDFM(cfg)
        net = model.net

        # Build one input batch that matches the requested backbone contract.
        if arch == "mlp":
            z = torch.randn(2, cfg.data.data_dim)
        else:
            z = torch.randn(
                2,
                cfg.data.image_channels,
                cfg.data.image_size,
                cfg.data.image_size,
            )
        t = torch.rand(z.shape[0])

        # Run one forward pass before printing the structural summary.
        with torch.no_grad():
            out = net(z, t)

        print(f"\n=== {arch.upper()} ===")

        # The summary and shape trace show the actual module tree selected by
        # factory resolution, not just the requested arch string.
        print_model_summary(model)
        print_forward_shapes(model, z, t)

        # `wrapped model class` is the sanity line for adapters such as UNet
        # and JiT: it tells us which implementation sits underneath the wrapper.
        print(f"net class            : {net.__class__.__module__}.{net.__class__.__name__}")
        if hasattr(net, "model"):
            print(
                f"wrapped model class  : {net.model.__class__.__module__}.{net.model.__class__.__name__}"
            )
        print(f"forward output shape : {tuple(out.shape)}")
