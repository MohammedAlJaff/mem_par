"""Image32 UNet preset for 128px runs."""

from prototyping.runners.config_presets import image32_base


def get_default_configs():
    config = image32_base.get_default_configs()
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
