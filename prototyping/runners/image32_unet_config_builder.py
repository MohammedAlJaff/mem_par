"""Configuration builder for the Image32 UNet runner."""

import copy
import os

from prototyping.runners.common import resolve_batch_cap
from prototyping.runners.config_presets import get_default_configs
from mp_config import clone_config, get_base_config
from prototyping.runners.metrics_image32 import DEFAULT_FIXED_TIMES


RUNNER_NAME = "image32_unet"
DEFAULT_STUDY_DIR = "outputs/prototype_runners/image32_unet"


def _default_wandb_entity(fallback):
    entity = os.getenv("WANDB_ENTITY", "").strip()
    return entity if entity else fallback


_UNET_PROFILES = {
    32: {
        "base_channels": 32,
        "channel_mult": (1, 2, 4),
        "attn_resolutions": (16, 8),
        "num_res_blocks": 2,
        "dropout": 0.1,
        "time_emb_dim": None,
    },
    128: {
        "base_channels": 64,
        "channel_mult": (1, 2, 4, 4, 8),
        "attn_resolutions": (16,),
        "num_res_blocks": 2,
        "dropout": 0.0,
        "time_emb_dim": None,
    },
    256: {
        "base_channels": 64,
        "channel_mult": (1, 2, 4, 4, 8, 8),
        "attn_resolutions": (16,),
        "num_res_blocks": 2,
        "dropout": 0.0,
        "time_emb_dim": None,
    },
}


DATASET_DEFAULTS = {
    "mnist": {
        "dataset": "mnist",
        "config_profile": "image32_32",
        "image_size": 32,
        "image_channels": 1,
        "resize_size": 0,
        "reference_split": "test",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[32],
    },
    "celeba32_grey": {
        "dataset": "celeba32_grey",
        "config_profile": "image32_32",
        "image_size": 32,
        "image_channels": 1,
        "resize_size": 0,
        "reference_split": "valid",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[32],
    },
    "flowers102": {
        "dataset": "flowers102",
        "config_profile": "image32_32",
        "image_size": 32,
        "image_channels": 3,
        "resize_size": 0,
        "reference_split": "val",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[32],
    },
    "flowers102_128": {
        "dataset": "flowers102",
        "config_profile": "image32_128",
        "image_size": 128,
        "image_channels": 3,
        "resize_size": 128,
        "reference_split": "val",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[128],
    },
    "flowers102_256": {
        "dataset": "flowers102",
        "config_profile": "image32_256",
        "image_size": 256,
        "image_channels": 3,
        "resize_size": 256,
        "reference_split": "val",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[256],
    },
    "afhq_v2": {
        "dataset": "afhq_v2",
        "config_profile": "image32_32",
        "image_size": 32,
        "image_channels": 3,
        "resize_size": 0,
        "reference_split": "test",
        "allow_same_source_reference": False,
        "dataset_subdir": "afhq_v2",
        **_UNET_PROFILES[32],
    },
    "ffhq128": {
        "dataset": "ffhq128",
        "config_profile": "image32_128",
        "image_size": 128,
        "image_channels": 3,
        "resize_size": 128,
        "reference_split": "seeded_partition",
        "allow_same_source_reference": True,
        "dataset_subdir": "ffhq/thumbnails128x128",
        **_UNET_PROFILES[128],
    },
}


DEFAULT_SETTINGS = {
    # Run identity and reproducibility.
    "device": "cpu",
    "out_dir": DEFAULT_STUDY_DIR,
    "run_name": RUNNER_NAME,
    "seed": 0,

    # Data/model recipe selection.
    "dataset": "celeba32_grey",
    "config_preset": None,
    "prediction_space": "v",
    "loss_space": "v",

    # Dataset sizes.
    # n_train: real images used for optimization.
    # n_eval: real held-out/reference images loaded for evaluation/baselines.
    "n_train": 128,
    "n_eval": 128,

    # Training schedule.
    "n_steps": 30000,
    "ckpt_every": 2000,
    "log_every": None,
    "batch_cap": 64,

    # Sampling/evaluation.
    # eval_samples: generated samples per checkpoint; also the size of z_init.pt.
    # sampling_steps: numerical solver steps used when turning z_init into images.
    "eval_samples": 64,
    "sampling_steps": 24,

    # Data location.
    "root": "data",
    "dataset_subdir": None,
    "download": False,

    # Dataset-profile fields. These are intentionally no longer CLI knobs.
    "train_split": "train",
    "reference_split": None,
    "allow_same_source_reference": None,
    "image_size": 32,
    "image_channels": 1,
    "resize_size": 0,
    "balance_train_classes": True,
    "balance_reference_classes": True,
    "loss_eval_times": DEFAULT_FIXED_TIMES,
    "base_channels": 32,
    "channel_mult": (1, 2, 4),
    "attn_resolutions": (16, 8),
    "num_res_blocks": 2,
    "dropout": 0.1,
    "time_emb_dim": None,

    # W&B basics. Entity/notes/strict are kept here for presets/notebooks, not CLI.
    "wandb": True,
    "wandb_mode": "online",
    "wandb_project": "",
    "wandb_entity": _default_wandb_entity("maj90"),
    "wandb_name": "",
    "wandb_group": "",
    "wandb_tags": "",
    "wandb_notes": "",
    "wandb_strict": False,

    # Optional visual trace during training.
    # Uses the same saved z_init.pt at step 0, each checkpoint, and final model.
    "generation_trace": False,
    "generation_trace_count": 8,
    "generation_trace_show": True,
}


_PRESET_OWNED_SETTING_KEYS = {
    "dataset_subdir",
    "train_split",
    "reference_split",
    "allow_same_source_reference",
    "image_size",
    "image_channels",
    "resize_size",
    "balance_train_classes",
    "balance_reference_classes",
    "loss_eval_times",
    "base_channels",
    "channel_mult",
    "attn_resolutions",
    "num_res_blocks",
    "dropout",
    "time_emb_dim",
    "wandb_entity",
    "wandb_notes",
    "wandb_strict",
}


_MODEL_SETTING_KEYS = {
    "base_channels",
    "channel_mult",
    "attn_resolutions",
    "num_res_blocks",
    "dropout",
    "time_emb_dim",
}


def get_default_settings():
    """Return canonical base defaults for image32 UNet runs."""
    return copy.deepcopy(DEFAULT_SETTINGS)


def list_datasets():
    """Return supported dataset keys for this runner."""
    return tuple(DATASET_DEFAULTS.keys())


def _coalesce(value, fallback=None):
    if isinstance(value, str) and not value.strip():
        return fallback
    if value is None:
        return fallback
    return value


def _raw_overrides_default(raw_settings, key):
    """Return True when a direct settings dict intentionally overrides a rare key."""
    if key not in raw_settings:
        return False
    if key not in DEFAULT_SETTINGS:
        return True
    return raw_settings[key] != DEFAULT_SETTINGS[key]


def _merge_settings(raw_settings, dataset_defaults):
    """
    Merge defaults without letting default-valued rare knobs clobber profiles.

    The CLI now exposes only sweep/prototype fields. Some notebook/tests still
    start from `default_settings()` and update a few rare fields directly; those
    explicit non-default updates remain honored.
    """
    merged = {**DEFAULT_SETTINGS, **dataset_defaults}
    for key, value in raw_settings.items():
        if (
            key in _PRESET_OWNED_SETTING_KEYS
            and key in DEFAULT_SETTINGS
            and value == DEFAULT_SETTINGS[key]
        ):
            continue
        merged[key] = value
    return merged


def _preset_value(defaults, section_name, key, fallback):
    section = getattr(defaults, section_name, None)
    if section is None:
        return fallback
    try:
        value = getattr(section, key)
    except AttributeError:
        return fallback
    return _coalesce(value, fallback)


def build_config(settings):
    """Build the canonical Image32 UNet config from merged run settings."""
    raw_settings = dict(settings)
    dataset = str(_coalesce(raw_settings.get("dataset"), DEFAULT_SETTINGS["dataset"]))
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown Image32 dataset: {dataset}")
    dataset_defaults = DATASET_DEFAULTS[dataset]
    settings = _merge_settings(raw_settings, dataset_defaults)

    explicit_preset = _coalesce(raw_settings.get("config_preset"), None) is not None
    preset_name = str(
        _coalesce(
            raw_settings.get("config_preset"),
            dataset_defaults.get("config_profile", "image32_32"),
        )
    )
    defaults = get_default_configs(preset_name)

    def data_setting(key):
        if _raw_overrides_default(raw_settings, key):
            return raw_settings[key]
        if explicit_preset:
            return _preset_value(defaults, "data", key, settings[key])
        return settings[key]

    def model_setting(key):
        if key not in _MODEL_SETTING_KEYS:
            raise KeyError(key)
        if _raw_overrides_default(raw_settings, key):
            return raw_settings[key]
        if explicit_preset:
            return _preset_value(defaults, "model", key, settings[key])
        return settings[key]

    config = clone_config(get_base_config())

    # Study metadata.
    config.study.name = RUNNER_NAME
    config.study.description = f"{dataset} Image32 GDFM/UNet prototype recipe."
    config.outputs.run_name = str(settings["run_name"])

    # Framework.
    config.framework.family = defaults.framework.family
    config.framework.prediction_space = str(settings["prediction_space"])
    config.framework.loss_space = str(settings["loss_space"])

    # Data.
    config.data.dataset = str(dataset_defaults["dataset"])
    config.data.root = str(settings["root"])
    config.data.dataset_subdir = str(
        _coalesce(data_setting("dataset_subdir"), dataset_defaults["dataset_subdir"])
    )
    config.data.download = bool(settings["download"])
    config.data.train_split = str(_coalesce(data_setting("train_split"), "train"))
    config.data.reference_split = str(
        _coalesce(data_setting("reference_split"), dataset_defaults["reference_split"])
    )
    config.data.allow_same_source_reference = bool(
        _coalesce(
            data_setting("allow_same_source_reference"),
            dataset_defaults["allow_same_source_reference"],
        )
    )
    config.data.n_train = int(settings["n_train"])
    config.data.n_eval = int(settings["n_eval"])
    config.data.image_size = int(data_setting("image_size"))
    config.data.image_channels = int(data_setting("image_channels"))
    config.data.resize_size = int(data_setting("resize_size"))
    config.data.flatten = False
    config.data.data_dim = (
        int(config.data.image_channels)
        * int(config.data.image_size)
        * int(config.data.image_size)
    )
    config.data.balance_train_classes = bool(data_setting("balance_train_classes"))
    config.data.balance_reference_classes = bool(
        data_setting("balance_reference_classes")
    )

    # Model / network.
    config.network.arch = defaults.model.arch
    config.network.base_channels = int(model_setting("base_channels"))
    config.network.channel_mult = tuple(
        _coalesce(model_setting("channel_mult"), defaults.model.channel_mult)
    )
    config.network.attn_resolutions = tuple(
        _coalesce(model_setting("attn_resolutions"), defaults.model.attn_resolutions)
    )
    config.network.num_res_blocks = int(model_setting("num_res_blocks"))
    config.network.dropout = float(model_setting("dropout"))
    config.network.time_emb_dim = model_setting("time_emb_dim")

    # Training.
    config.training.seed = int(settings["seed"])
    config.training.n_steps = int(settings["n_steps"])
    config.training.ckpt_every = int(settings["ckpt_every"])
    log_every = settings.get("log_every")
    if log_every is None:
        config.training.log_every = max(1, min(500, int(config.training.ckpt_every)))
    else:
        config.training.log_every = max(1, int(log_every))
    batch_cap = resolve_batch_cap(settings)
    config.training.batch_cap = batch_cap
    config.training.batch_size = batch_cap
    config.training.optimizer = defaults.training.optimizer
    lr = raw_settings["lr"] if _raw_overrides_default(raw_settings, "lr") else None
    config.training.lr = float(_coalesce(lr, defaults.training.lr))
    weight_decay = (
        raw_settings["weight_decay"]
        if _raw_overrides_default(raw_settings, "weight_decay")
        else None
    )
    config.training.weight_decay = float(
        _coalesce(weight_decay, defaults.training.weight_decay)
    )

    # Evaluation.
    config.evaluation.n_samples = int(
        _coalesce(settings["eval_samples"], defaults.evaluation.n_samples)
    )
    config.evaluation.batch_size = batch_cap
    config.evaluation.sample_panel_count = min(
        int(_coalesce(defaults.evaluation.sample_panel_count, 8)),
        int(config.evaluation.n_samples),
    )
    loss_eval_times = (
        raw_settings["loss_eval_times"]
        if _raw_overrides_default(raw_settings, "loss_eval_times")
        else defaults.evaluation.loss_eval_times
    )
    config.evaluation.loss_eval_times = tuple(float(item) for item in loss_eval_times)
    config.evaluation.loss_eval_batch_size = int(
        _coalesce(getattr(defaults.evaluation, "loss_eval_batch_size", 0), 0)
    )
    config.evaluation.chunk_size = max(
        int(_coalesce(defaults.evaluation.chunk_size, 128)),
        int(config.evaluation.n_samples),
        int(config.data.n_eval),
        int(config.data.n_train),
    )

    # Sampling.
    config.sampling.method = defaults.sampling.method
    config.sampling.n_steps = int(
        _coalesce(settings["sampling_steps"], defaults.sampling.n_steps)
    )

    # Generation trace.
    config.generation_trace.enabled = bool(settings["generation_trace"])
    config.generation_trace.num_images = int(settings["generation_trace_count"])
    config.generation_trace.show = bool(settings["generation_trace_show"])

    # Tracking.
    config.tracking.console.summary_every_steps = int(config.training.log_every)
    config.tracking.console.summary_every_seconds = float(
        defaults.tracking.console.summary_every_seconds
    )
    config.tracking.wandb.enabled = bool(settings["wandb"])
    config.tracking.wandb.mode = str(settings["wandb_mode"])
    config.tracking.wandb.project = str(settings["wandb_project"])
    wandb_entity = (
        raw_settings["wandb_entity"]
        if _raw_overrides_default(raw_settings, "wandb_entity")
        else defaults.tracking.wandb.entity
    )
    wandb_notes = (
        raw_settings["wandb_notes"]
        if _raw_overrides_default(raw_settings, "wandb_notes")
        else defaults.tracking.wandb.notes
    )
    wandb_strict = (
        raw_settings["wandb_strict"]
        if _raw_overrides_default(raw_settings, "wandb_strict")
        else defaults.tracking.wandb.strict
    )
    config.tracking.wandb.entity = str(wandb_entity)
    config.tracking.wandb.notes = str(wandb_notes)
    config.tracking.wandb.name = str(settings["wandb_name"])
    config.tracking.wandb.group = str(settings["wandb_group"])
    config.tracking.wandb.tags = [
        item.strip()
        for item in str(settings["wandb_tags"]).split(",")
        if item.strip()
    ]
    config.tracking.wandb.strict = bool(wandb_strict)
    return config
