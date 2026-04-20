"""Image32 (32x32) UNet config preset."""

from prototyping.runners.config_presets import image32_base


def get_default_configs():
    config = image32_base.get_default_configs()
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
