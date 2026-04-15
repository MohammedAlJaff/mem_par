from mp_config import get_base_config


def get_config():
    config = get_base_config()
    config.study.name = "mnist_n4_diagonal"
    config.study.description = "MNIST diagonal comparison at N=4."

    config.framework.family = "fm"
    config.framework.prediction_space = "x"
    config.framework.loss_space = "x"

    config.network.arch = "mlp"
    config.network.hidden_dim = 256
    config.network.n_layers = 3
    config.network.time_emb_dim = 64

    config.data.dataset = "mnist"
    config.data.root = "tmp/data"
    config.data.download = False
    config.data.n_train = 4
    config.data.n_eval = 128
    config.data.data_dim = 784
    config.data.image_size = 28
    config.data.image_channels = 1
    config.data.flatten = True

    config.training.optimizer = "adam"
    config.training.n_steps = 500
    config.training.batch_size = 4
    config.training.lr = 3.0e-4
    config.training.ckpt_every = 100
    config.training.log_every = 10

    config.evaluation.n_samples = 16
    config.evaluation.sample_panel_count = 8

    config.sampling.method = "heun"
    config.sampling.n_steps = 50

    config.outputs.run_name = "mnist_n4_diagonal"
    return config
