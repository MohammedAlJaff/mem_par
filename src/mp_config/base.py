"""
Canonical config helpers for first-party studies.

This module owns the baseline `ConfigDict` shape used by the active study
surface. It also owns the small conversion and override helpers needed to:
- clone configs safely
- serialize them for checkpoints and run artifacts
- apply dotted command-line-style overrides without rewriting the config by hand
"""

from ml_collections import ConfigDict


def get_base_config():
    """
    Build the canonical nested config used by the first-party study stack.

    The base config is intentionally broad enough to cover the active study
    families:
        - framework and schedule choices for GDFM
        - backbone hyperparameters
        - dataset and split settings
        - training, evaluation, sampling, and output settings

    A first-year PhD should be able to treat this function as the map of the
    whole study system: every major subsystem gets one nested section, and new
    studies are expected to modify values inside that shape rather than invent
    a parallel config layout.
    """
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
    data.balance_train_classes = False
    data.balance_reference_classes = False
    data.allow_same_source_reference = False
    data.n_clusters = 10
    data.sigma_z = 0.5
    data.sigma_eta = 1.0e-3
    data.geometry_seed = 0
    data.reference_split = "valid"

    training = config.training = ConfigDict()
    training.seed = 0
    training.n_steps = 200
    training.batch_size = 64
    training.lr = 1.0e-3
    training.weight_decay = 0.0
    training.optimizer = "adam"
    training.momentum = 0.9
    training.ckpt_every = 50
    training.log_every = 10
    training.show_progress = True

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

    tracking = config.tracking = ConfigDict()

    tracking_console = tracking.console = ConfigDict()
    tracking_console.enabled = True
    tracking_console.summary_every_steps = int(training.log_every)
    tracking_console.summary_every_seconds = 60.0
    tracking_console.stage_banners = True
    tracking_console.tee_to_attempt_log = True

    tracking_wandb = tracking.wandb = ConfigDict()
    tracking_wandb.enabled = False
    tracking_wandb.mode = "offline"
    tracking_wandb.project = "mem_par"
    tracking_wandb.entity = ""
    tracking_wandb.dir = "logs/wandb"
    tracking_wandb.tags = []
    tracking_wandb.notes = ""
    tracking_wandb.strict = False

    return config


def config_to_dict(value):
    """
    Recursively convert `ConfigDict`-style containers into plain Python types.

    We use this at the artifact boundary: once a run is on disk, the config
    should be inspectable by humans and small scripts without importing
    `ml_collections`.
    """
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
    """
    Deep-clone a config by round-tripping it through plain Python containers.
    """
    return ConfigDict(config_to_dict(config))


def _set_dotted(config, dotted_key, value):
    """
    Set one nested config field using a dotted key such as `training.lr`.
    """
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor:
            cursor[part] = ConfigDict()
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(config, overrides):
    """
    Apply a dict of dotted-key overrides onto an existing config in place.

    This is the bridge between the clean nested config shape and the reality of
    CLI experimentation, where users often want to change one or two fields
    without writing a new config file.
    """
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        _set_dotted(config, dotted_key, value)
    return config


if __name__ == "__main__":
    print("mp_config.base example")

    # Start from the canonical base config, then make an editable copy.
    config = get_base_config()
    cloned = clone_config(config)

    # Apply a few dotted-path overrides, exactly as a study script would.
    apply_overrides(
        cloned,
        {
            "study.name": "demo_study",
            "framework.prediction_space": "v",
            "training.lr": 1.0e-3,
        },
    )

    # Convert back to a plain dict so the structure is easy to inspect or save.
    payload = config_to_dict(cloned)

    # These prints check that cloning, overrides, and dict export all worked.
    print(f"base study name         : {config.study.name}")
    print(f"cloned study name       : {cloned.study.name}")
    print(f"prediction_space        : {cloned.framework.prediction_space}")
    print(f"training.lr             : {cloned.training.lr}")
    print(f"top-level config keys   : {sorted(payload.keys())}")
