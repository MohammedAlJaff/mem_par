from ml_collections import ConfigDict


def get_base_config():
    config = ConfigDict()

    study = config.study = ConfigDict()
    study.name = "unnamed_study"
    study.description = "Canonical outer-shell study."

    framework = config.framework = ConfigDict()
    framework.family = "fm"
    framework.prediction_space = "x"
    framework.loss_space = "x"

    schedule = config.schedule = ConfigDict()
    schedule.t_min = 1.0e-3
    schedule.t_max = 0.999
    schedule.denom_eps = 0.05
    schedule.beta_min = 0.1
    schedule.beta_max = 20.0
    schedule.time_sampling = "uniform"
    schedule.logit_mu = 0.0
    schedule.logit_sigma = 1.0

    network = config.network = ConfigDict()
    network.arch = "mlp"
    network.hidden_dim = 128
    network.n_layers = 3
    network.time_emb_dim = 64
    network.base_channels = 64
    network.channel_mult = (1, 2, 4, 4)
    network.attn_resolutions = (8,)
    network.num_res_blocks = 2
    network.dropout = 0.0
    network.patch_size = 16
    network.depth = 12
    network.heads = 12
    network.bottleneck_dim = 128
    network.proj_dropout = 0.0
    network.attn_dropout = 0.0
    network.in_context_len = 32
    network.in_context_start = 4

    data = config.data = ConfigDict()
    data.dataset = "gmm_subspace"
    data.root = "tmp/data"
    data.download = False
    data.n_train = 256
    data.n_eval = 256
    data.data_dim = 256
    data.intrinsic_dim = 4
    data.image_size = 0
    data.image_channels = 0
    data.flatten = False
    data.n_clusters = 10
    data.sigma_z = 0.5
    data.sigma_eta = 1.0e-3
    data.geometry_seed = 0
    data.reference_split = "valid"

    training = config.training = ConfigDict()
    training.seed = 0
    training.n_steps = 200
    training.batch_size = 64
    training.lr = 3.0e-4
    training.weight_decay = 0.0
    training.optimizer = "adam"
    training.momentum = 0.9
    training.ckpt_every = 50
    training.log_every = 10

    evaluation = config.evaluation = ConfigDict()
    evaluation.n_samples = 64
    evaluation.batch_size = 64
    evaluation.ratio_threshold = 0.5
    evaluation.sample_panel_count = 8
    evaluation.chunk_size = 256

    sampling = config.sampling = ConfigDict()
    sampling.method = "heun"
    sampling.n_steps = 50

    outputs = config.outputs = ConfigDict()
    outputs.run_name = "run"

    return config


def config_to_dict(value):
    if isinstance(value, ConfigDict):
        return {key: config_to_dict(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: config_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [config_to_dict(item) for item in value]
    if isinstance(value, list):
        return [config_to_dict(item) for item in value]
    return value


def clone_config(config):
    return ConfigDict(config_to_dict(config))


def _set_dotted(config, dotted_key, value):
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor:
            cursor[part] = ConfigDict()
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(config, overrides):
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        _set_dotted(config, dotted_key, value)
    return config
