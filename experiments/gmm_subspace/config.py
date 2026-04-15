from mp_config import get_base_config


def get_config():
    config = get_base_config()
    config.study.name = "gmm_subspace"
    config.study.description = "High-dimensional GMM on a linear subspace."

    config.framework.family = "fm"
    config.framework.prediction_space = "x"
    config.framework.loss_space = "x"

    config.network.arch = "mlp"
    config.network.hidden_dim = 256
    config.network.n_layers = 3
    config.network.time_emb_dim = 64

    config.data.dataset = "gmm_subspace"
    config.data.data_dim = 256
    config.data.intrinsic_dim = 4
    config.data.n_train = 256
    config.data.n_eval = 512
    config.data.n_clusters = 10
    config.data.sigma_z = 0.5
    config.data.sigma_eta = 1.0e-3
    config.data.flatten = False

    config.training.optimizer = "adam"
    config.training.n_steps = 1000
    config.training.batch_size = 64
    config.training.lr = 3.0e-4
    config.training.ckpt_every = 200
    config.training.log_every = 20

    config.evaluation.n_samples = 64
    config.evaluation.sample_panel_count = 8

    config.sampling.method = "heun"
    config.sampling.n_steps = 50

    config.outputs.run_name = "gmm_subspace"
    return config
