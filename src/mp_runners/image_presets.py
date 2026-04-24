"""
Image UNet presets used by the image runner.

Presets describe model scale and sampling defaults for common image sizes.
Dataset choice, train/reference counts, W&B names, and output paths are applied
later by `mp_runners.image_config`.
"""

import os
from typing import Callable, Dict

from ml_collections import ConfigDict


DEFAULT_PRESET = "image32_32"


def _default_wandb_entity(fallback):
    """Return the environment W&B entity or a fallback."""
    return os.getenv("WANDB_ENTITY", "").strip() or fallback


def _base_config():
    """
    Return shared Image UNet defaults before resolution-specific overrides.

    The config is intentionally plain so researchers can inspect all defaults in
    one place.
    """
    config = ConfigDict()

    config.study = ConfigDict()
    config.study.name = "image_runner"
    config.study.description = "Image UNet runner."

    config.framework = ConfigDict()
    config.framework.family = "fm"
    config.framework.prediction_space = "v"
    config.framework.loss_space = "v"

    config.model = ConfigDict()
    config.model.arch = "unet"
    config.model.base_channels = 32
    config.model.channel_mult = (1, 2, 4)
    config.model.attn_resolutions = (16, 8)
    config.model.num_res_blocks = 2
    config.model.dropout = 0.1
    config.model.time_emb_dim = None

    config.data = ConfigDict()
    config.data.image_size = 32
    config.data.resize_size = 0

    config.training = ConfigDict()
    config.training.seed = 0
    config.training.n_training_steps = 30000
    config.training.ckpt_every = 2000
    config.training.log_every = 200
    config.training.optimizer = "adam"
    config.training.lr = 1.0e-4
    config.training.weight_decay = 0.0

    config.evaluation = ConfigDict()
    config.evaluation.n_samples = 64
    config.evaluation.sample_panel_count = 8
    config.evaluation.chunk_size = 128
    config.evaluation.loss_eval_times = (0.5, 0.75)
    config.evaluation.loss_eval_batch_size = 0

    config.sampling = ConfigDict()
    config.sampling.method = "heun"
    config.sampling.n_steps = 24

    config.tracking = ConfigDict()
    config.tracking.wandb = ConfigDict()
    config.tracking.wandb.enabled = True
    config.tracking.wandb.mode = "online"
    config.tracking.wandb.project = ""
    config.tracking.wandb.entity = _default_wandb_entity("maj90")
    config.tracking.wandb.notes = ""
    config.tracking.wandb.name = ""
    config.tracking.wandb.group = ""
    config.tracking.wandb.tags = []
    config.tracking.wandb.strict = False

    config.tracking.console = ConfigDict()
    config.tracking.console.summary_every_seconds = 60.0
    return config


def _image32_32_config():
    """Return the 32px image preset."""
    config = _base_config()
    config.model.base_channels = 32
    config.model.channel_mult = (1, 2, 4)
    config.model.attn_resolutions = (16, 8)
    config.model.num_res_blocks = 2
    config.model.dropout = 0.1
    config.model.time_emb_dim = None
    config.data.image_size = 32
    config.data.resize_size = 0
    config.sampling.n_steps = 24
    return config


def _image32_64_config():
    """Return the 64px image preset."""
    config = _base_config()
    config.model.base_channels = 64
    config.model.channel_mult = (1, 2, 4, 4)
    config.model.attn_resolutions = (16,)
    config.model.num_res_blocks = 2
    config.model.dropout = 0.0
    config.model.time_emb_dim = None
    config.data.image_size = 64
    config.data.resize_size = 64
    config.sampling.n_steps = 24
    return config


def _image32_128_config():
    """Return the 128px image preset."""
    config = _base_config()
    config.model.base_channels = 64
    config.model.channel_mult = (1, 2, 4, 4, 8)
    config.model.attn_resolutions = (16,)
    config.model.num_res_blocks = 2
    config.model.dropout = 0.0
    config.model.time_emb_dim = None
    config.data.image_size = 128
    config.data.resize_size = 128
    config.sampling.n_steps = 24
    return config


def _image32_256_config():
    """Return the 256px image preset."""
    config = _base_config()
    config.model.base_channels = 64
    config.model.channel_mult = (1, 2, 4, 4, 8, 8)
    config.model.attn_resolutions = (16,)
    config.model.num_res_blocks = 2
    config.model.dropout = 0.0
    config.model.time_emb_dim = None
    config.data.image_size = 256
    config.data.resize_size = 256
    config.sampling.n_steps = 24
    return config


_PRESET_MAP: Dict[str, Callable[[], object]] = {
    "image32_32": _image32_32_config,
    "image32_64": _image32_64_config,
    "image32_128": _image32_128_config,
    "image32_256": _image32_256_config,
}


def list_presets():
    """Return sorted preset names."""
    return sorted(_PRESET_MAP.keys())


def get_default_configs(preset_name=DEFAULT_PRESET):
    """Return the requested preset config."""
    if preset_name not in _PRESET_MAP:
        raise ValueError(f"Unknown Image32 UNet preset: {preset_name}")
    return _PRESET_MAP[preset_name]()


if __name__ == "__main__":
    print("mp_runners.image_presets example")
    print("available presets:", ", ".join(list_presets()))
