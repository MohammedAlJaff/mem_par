"""Image32 (32x32) UNet config preset."""

from prototyping.runners.config_presets import image32_base


def get_default_configs():
    config = image32_base.get_default_configs()
    config.model.base_channels = 64
    config.model.channel_mult = (1, 2, 4, 4)
    config.model.attn_resolutions = (8,)
    config.model.num_res_blocks = 2
    config.sampling.n_steps = 24
    return config
