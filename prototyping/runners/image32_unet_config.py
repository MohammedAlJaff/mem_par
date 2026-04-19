"""Configuration builder for the Image32 UNet runner."""

from prototyping.runners.common import resolve_batch_cap
from prototyping.runners.default_knobs.image32_unet import (
    DATASET_KNOB_RECIPES as DATASET_DEFAULTS,
    default_knobs as _default_knobs,
)
from prototyping.runners.image32_unet_configs import get_default_configs
from mp_config import clone_config, get_base_config


RUNNER_NAME = "image32_unet"


def _coerce_knob(value, fallback=None):
    if value is None:
        return fallback
    return value


def build_config(knobs):
    """Build the canonical Image32 UNet config from merged knobs."""
    knobs = dict(knobs)
    dataset = str(knobs.get("dataset", "celeba32_grey"))
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown Image32 dataset: {dataset}")
    dataset_defaults = DATASET_DEFAULTS[dataset]
    knobs = {**_default_knobs(), **dataset_defaults, **knobs}

    config_name = str(
        knobs.get("config_preset", dataset_defaults.get("config_profile", "image32_32"))
    )
    defaults = get_default_configs(config_name)

    config = clone_config(get_base_config())

    # Study metadata.
    config.study.name = RUNNER_NAME
    config.study.description = f"{dataset} Image32 GDFM/UNet prototype recipe."
    config.outputs.run_name = str(knobs["run_name"])

    # Framework.
    config.framework.family = defaults.framework.family
    config.framework.prediction_space = str(knobs["prediction_space"])
    config.framework.loss_space = str(knobs["loss_space"])

    # Data. (We keep this here because `build_data_bundle` depends on it.)
    config.data.dataset = str(dataset_defaults["dataset"])
    config.data.root = str(knobs["root"])
    config.data.dataset_subdir = str(
        knobs["dataset_subdir"]
        if knobs["dataset_subdir"] is not None
        else dataset_defaults["dataset_subdir"]
    )
    config.data.download = bool(knobs["download"])
    config.data.train_split = str(knobs["train_split"])
    config.data.reference_split = str(
        knobs["reference_split"]
        if knobs["reference_split"] is not None
        else dataset_defaults["reference_split"]
    )
    config.data.allow_same_source_reference = bool(
        knobs["allow_same_source_reference"]
        if knobs["allow_same_source_reference"] is not None
        else dataset_defaults["allow_same_source_reference"]
    )
    config.data.n_train = int(knobs["n_train"])
    config.data.n_eval = int(knobs["n_eval"])
    config.data.image_size = int(knobs["image_size"])
    config.data.image_channels = int(knobs["image_channels"])
    config.data.resize_size = int(knobs["resize_size"])
    config.data.flatten = False
    config.data.data_dim = (
        int(knobs["image_channels"])
        * int(knobs["image_size"])
        * int(knobs["image_size"])
    )
    config.data.balance_train_classes = bool(knobs["balance_train_classes"])
    config.data.balance_reference_classes = bool(knobs["balance_reference_classes"])

    # Model / network.
    config.network.arch = defaults.model.arch
    config.network.base_channels = int(knobs["base_channels"])
    config.network.channel_mult = tuple(_coerce_knob(knobs["channel_mult"], defaults.model.channel_mult))
    config.network.attn_resolutions = tuple(
        _coerce_knob(knobs["attn_resolutions"], defaults.model.attn_resolutions)
    )
    config.network.num_res_blocks = int(knobs["num_res_blocks"])
    config.network.dropout = float(knobs["dropout"])
    config.network.time_emb_dim = knobs["time_emb_dim"]

    # Training.
    config.training.seed = int(knobs["seed"])
    config.training.n_steps = int(knobs["n_steps"])
    config.training.ckpt_every = int(knobs["ckpt_every"])
    config.training.log_every = max(1, min(500, int(config.training.ckpt_every)))
    batch_cap = resolve_batch_cap(knobs)
    config.training.batch_cap = batch_cap
    config.training.batch_size = batch_cap
    config.training.optimizer = defaults.training.optimizer
    config.training.lr = float(_coerce_knob(knobs.get("lr"), defaults.training.lr))
    config.training.weight_decay = float(
        _coerce_knob(knobs.get("weight_decay"), defaults.training.weight_decay)
    )

    # Evaluation.
    config.evaluation.n_samples = int(_coerce_knob(knobs["eval_samples"], defaults.evaluation.n_samples))
    config.evaluation.batch_size = batch_cap
    config.evaluation.sample_panel_count = min(
        int(_coerce_knob(defaults.evaluation.sample_panel_count, 8)),
        int(config.evaluation.n_samples),
    )
    config.evaluation.loss_eval_times = tuple(
        float(item) for item in _coerce_knob(
            knobs["loss_eval_times"], defaults.evaluation.loss_eval_times
        )
    )
    config.evaluation.chunk_size = max(
        int(_coerce_knob(defaults.evaluation.chunk_size, 128)),
        int(config.evaluation.n_samples),
        int(config.data.n_eval),
        int(config.data.n_train),
    )

    # Sampling.
    config.sampling.method = defaults.sampling.method
    config.sampling.n_steps = int(_coerce_knob(knobs["sampling_steps"], defaults.sampling.n_steps))

    # Tracking.
    config.tracking.console.summary_every_steps = int(config.training.log_every)
    config.tracking.console.summary_every_seconds = float(
        defaults.tracking.console.summary_every_seconds
    )
    config.tracking.wandb.enabled = bool(knobs["wandb"])
    config.tracking.wandb.mode = str(knobs["wandb_mode"])
    config.tracking.wandb.project = str(knobs["wandb_project"])
    config.tracking.wandb.entity = str(knobs["wandb_entity"])
    config.tracking.wandb.notes = str(knobs["wandb_notes"])
    config.tracking.wandb.name = str(knobs["wandb_name"])
    config.tracking.wandb.group = str(knobs["wandb_group"])
    config.tracking.wandb.tags = [
        item.strip()
        for item in str(knobs["wandb_tags"]).split(",")
        if item.strip()
    ]
    config.tracking.wandb.strict = bool(knobs["wandb_strict"])
    return config
