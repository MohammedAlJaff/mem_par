from mp_config import get_base_config


def get_config():
    config = get_base_config()
    config.study.name = "celeba32_grey"
    config.study.description = "CelebA 32x32 greyscale image study."

    config.framework.family = "vp"
    config.framework.prediction_space = "eps"
    config.framework.loss_space = "eps"

    config.network.arch = "unet"
    config.network.base_channels = 32
    config.network.channel_mult = (1, 2, 4, 4)
    config.network.attn_resolutions = (8,)
    config.network.num_res_blocks = 2
    config.network.time_emb_dim = 128

    config.data.dataset = "celeba32_grey"
    config.data.root = "tmp/data"
    config.data.download = False
    config.data.n_train = 128
    config.data.n_eval = 256
    config.data.image_size = 32
    config.data.image_channels = 1
    config.data.flatten = False
    config.data.reference_split = "valid"

    config.training.optimizer = "sgd"
    config.training.momentum = 0.9
    config.training.n_steps = 400
    config.training.batch_size = 64
    config.training.lr = 1.0e-2
    config.training.ckpt_every = 100
    config.training.log_every = 20

    config.evaluation.n_samples = 16
    config.evaluation.sample_panel_count = 8

    config.sampling.method = "heun"
    config.sampling.n_steps = 50

    config.outputs.run_name = "celeba32_grey"
    return config
