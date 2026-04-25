"""
Configuration builder for the image UNet runner.

The runner receives simple CLI/settings values. This module turns them into the
canonical nested config used by data loading, model construction, training,
evaluation, runtime precision, and tracking.
"""

import argparse
import copy
import os
import re

from mp_config import clone_config, get_base_config
from mp_metrics.loss_eval import DEFAULT_LOSS_EVAL_TIMES
from mp_runners.common import resolve_batch_cap
from mp_training.checkpoint_schedule import (
    parse_checkpoint_steps,
    resolve_checkpoint_steps,
)
from mp_tracking import resolve_wandb_identity


RUNNER_NAME = "image_train_runner"
DEFAULT_STUDY_DIR = "outputs/runners/image_train_runner"
MODEL_CHOICES = ("unet",)
DEFAULT_IMAGE_LR = 1.0e-4
DEFAULT_IMAGE_WEIGHT_DECAY = 0.0
DEFAULT_IMAGE_OPTIMIZER = "adam"
DEFAULT_CONSOLE_SUMMARY_SECONDS = 60.0
DEFAULT_EVAL_SAMPLE_BANK_SIZE = 1024


def _default_wandb_entity(fallback):
    """Return the environment W&B entity or a fallback."""
    entity = os.getenv("WANDB_ENTITY", "").strip()
    return entity if entity else fallback


_LEGACY_BROAD_WANDB_PROJECTS = {
    "memparam-image32-unet",
    "image32-unet",
}


def _wandb_slug(value, fallback):
    """Normalize a W&B slug component."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-")
    return text or fallback


def _dataset_model_wandb_project(dataset_key, image_size, model_arch):
    """Build the default W&B project name for a dataset/size/model pair."""
    dataset = _wandb_slug(dataset_key, "dataset")
    size = _wandb_slug(image_size, "size")
    model = _wandb_slug(model_arch, "model")
    return f"memparam-{dataset}-{size}-{model}"


def _resolve_wandb_project(raw_project, dataset_key, image_size, model_arch):
    """Resolve explicit, legacy, and automatic W&B project names."""
    requested = str(raw_project or "").strip()
    canonical = _dataset_model_wandb_project(dataset_key, image_size, model_arch)
    if not requested or requested.lower() in _LEGACY_BROAD_WANDB_PROJECTS:
        return canonical
    if requested.lower() in {"auto", "dataset-model"}:
        return canonical
    return requested


_UNET_PROFILES = {
    32: {
        "base_channels": 32,
        "channel_mult": (1, 2, 4),
        "attn_resolutions": (16, 8),
        "num_res_blocks": 2,
        "dropout": 0.1,
        "time_emb_dim": None,
    },
    64: {
        "base_channels": 64,
        "channel_mult": (1, 2, 4, 4),
        "attn_resolutions": (16,),
        "num_res_blocks": 2,
        "dropout": 0.0,
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


_KNOWN_SPLIT_SIZES = {
    "mnist": {"train": 60000, "test": 10000},
    "celeba32_grey": {"train": 162770, "valid": 19867, "test": 19962},
    "flowers102": {"train": 1020, "val": 1020, "valid": 1020, "test": 6149},
}


DATASET_DEFAULTS = {
    "mnist": {
        "dataset": "mnist",
        "recipe": "default",
        "image_size": 32,
        "image_channels": 1,
        "resize_size": 0,
        "reference_split": "test",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[32],
    },
    "celeba": {
        "dataset": "celeba32_grey",
        "recipe": "bonnaire_celeba32",
        "image_size": 32,
        "image_channels": 1,
        "resize_size": 0,
        "reference_split": "valid",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[32],
    },
    "flowers": {
        "dataset": "flowers102",
        "recipe": "default",
        "image_size": 32,
        "image_channels": 3,
        "resize_size": 0,
        # Flowers102's official train split has only 1,020 images.
        # For generative runs we train on the large official test split and use
        # the small official train split as the held-out/reference set.
        "train_split": "test",
        "reference_split": "train",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[32],
    },
    "afhq": {
        "dataset": "afhq_v2",
        "recipe": "default",
        "image_size": 32,
        "image_channels": 3,
        "resize_size": 0,
        "reference_split": "test",
        "allow_same_source_reference": False,
        "dataset_subdir": "afhq_v2",
        **_UNET_PROFILES[32],
    },
    "ffhq": {
        "dataset": "ffhq128",
        "recipe": "default",
        "image_size": 128,
        "image_channels": 3,
        "resize_size": 128,
        "reference_split": "seeded_partition",
        "allow_same_source_reference": True,
        "dataset_subdir": "ffhq/thumbnails128x128",
        **_UNET_PROFILES[128],
    },
}


PUBLIC_DATASETS = ("mnist", "celeba", "flowers", "afhq", "ffhq")
SCALABLE_IMAGE_DATASETS = {"mnist", "flowers", "afhq"}


DEFAULT_SETTINGS = {
    # Run identity and reproducibility.
    "device": "cpu",
    "out_dir": DEFAULT_STUDY_DIR,
    "run_name": RUNNER_NAME,
    "seed": 0,

    # Data/model runner selection.
    "dataset": "celeba",
    "model": "unet",
    "prediction_space": "v",
    "loss_space": "v",

    # Dataset sizes.
    # n_train: real images used for optimization.
    # n_test: real held-out/reference images loaded for evaluation/baselines.
    "n_train": 128,
    "n_test": 128,

    # Training schedule.
    "n_training_steps": 30000,
    "ckpt_every": 2000,
    "ckpt_schedule": "bonnaire-1m",
    "ckpt_steps": "",
    "log_every": None,
    "batch_cap": 64,
    "lr": DEFAULT_IMAGE_LR,
    "weight_decay": DEFAULT_IMAGE_WEIGHT_DECAY,

    # Runtime numerics.
    # precision: fp32 preserves current behavior; tf32/bf16 are CUDA speed paths.
    "precision": "fp32",
    "data_summary": False,

    # Sampling/monitoring.
    # eval_sample_bank_size: fixed-z bank saved for post-hoc checkpoint eval.
    # generation_monitor_samples: generated fixed-z samples shown in monitors.
    # sampling_steps: numerical solver steps used when turning z_init into images.
    "eval_sample_bank_size": DEFAULT_EVAL_SAMPLE_BANK_SIZE,
    "generation_monitor_samples": 64,
    "sampling_steps": 24,

    # Data location.
    "root": "data",
    "dataset_subdir": None,
    "download": False,

    # Dataset-profile fields. These are intentionally no longer CLI settings.
    "train_split": "train",
    "reference_split": None,
    "allow_same_source_reference": None,
    "image_size": None,
    "image_channels": 1,
    "resize_size": 0,
    "balance_train_classes": True,
    "balance_reference_classes": True,
    "loss_eval_times": DEFAULT_LOSS_EVAL_TIMES,
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

    # Optional visual monitor during training.
    # Uses the same saved z_init.pt at step 0, each checkpoint, and final model.
    "generation_monitor": False,
    "generation_monitor_show": True,
    "compact_model_summary": True,

    # Always-on image train-time monitor. This is not post-hoc checkpoint evaluation.
    "random_projected_f_mem_samples": 512,
    "random_projected_f_mem_resize": 16,
    "random_projected_f_mem_proj_dim": 256,
    "random_projected_f_mem_proj_seed": None,
}


_PROFILE_OWNED_SETTING_KEYS = {
    "dataset_subdir",
    "train_split",
    "reference_split",
    "allow_same_source_reference",
    "recipe",
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
    """Return canonical base defaults for image UNet train runs."""
    return copy.deepcopy(DEFAULT_SETTINGS)


def list_datasets():
    """Return supported dataset keys for this runner."""
    return PUBLIC_DATASETS


def list_models():
    """Return supported image model names."""
    return MODEL_CHOICES


def list_image_sizes():
    """Return supported automatic image sizes."""
    return tuple(sorted(_UNET_PROFILES.keys()))


def _coalesce(value, fallback=None):
    """Return fallback for None or empty strings."""
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
    Merge defaults without letting default-valued rare settings override profiles.

    The CLI now exposes only sweep fields. Some notebook/tests still
    start from `default_settings()` and update a few rare fields directly; those
    explicit non-default updates remain honored.
    """
    merged = {**DEFAULT_SETTINGS, **dataset_defaults}
    for key, value in raw_settings.items():
        if (
            key in _PROFILE_OWNED_SETTING_KEYS
            and key in DEFAULT_SETTINGS
            and value == DEFAULT_SETTINGS[key]
        ):
            continue
        merged[key] = value
    return merged


def _resolve_model(raw_model):
    """Resolve the user-facing model name."""
    model = str(_coalesce(raw_model, DEFAULT_SETTINGS["model"])).lower()
    if model not in MODEL_CHOICES:
        raise ValueError(
            f"Unknown image model: {model}. Supported models: {', '.join(MODEL_CHOICES)}"
        )
    return model


def _resolve_requested_image_size(raw_size):
    """Return the requested image size, or None when the dataset default should apply."""
    if raw_size is None:
        return None
    if isinstance(raw_size, str) and not raw_size.strip():
        return None
    size = int(raw_size)
    if size < 1:
        raise ValueError(f"image_size must be positive, got {size}")
    return size


def _profile_key_for_size(dataset, image_size):
    """Return the public dataset key for a requested size."""
    return dataset


def _sized_dataset_defaults(dataset, image_size):
    """Build dataset defaults for a supported base dataset at a requested size."""
    image_size = int(image_size)
    defaults = copy.deepcopy(DATASET_DEFAULTS[dataset])
    if image_size in _UNET_PROFILES:
        defaults.update(_UNET_PROFILES[image_size])
    defaults["image_size"] = image_size
    defaults["resize_size"] = 0 if image_size == 32 else image_size
    return defaults


def _resolve_dataset_profile(raw_settings):
    """
    Resolve dataset, image size, and model into one concrete dataset profile.

    This is the main convenience path for the image runner: callers can pass a
    public dataset such as `flowers`, an image size such as `64`, and a model
    such as `unet`; the runner chooses the matching data/model defaults.
    """
    model = _resolve_model(raw_settings.get("model", DEFAULT_SETTINGS["model"]))
    raw_dataset = str(_coalesce(raw_settings.get("dataset"), DEFAULT_SETTINGS["dataset"]))
    dataset = raw_dataset
    if dataset not in DATASET_DEFAULTS:
        choices = ", ".join(PUBLIC_DATASETS)
        raise ValueError(
            f"Unknown image runner dataset: {raw_dataset}. "
            f"Use one of: {choices}"
        )

    image_size = _resolve_requested_image_size(raw_settings.get("image_size"))
    defaults = DATASET_DEFAULTS[dataset]
    if image_size is None:
        return dataset, defaults, model

    default_size = int(defaults["image_size"])
    if int(image_size) == default_size:
        return dataset, defaults, model

    if dataset not in SCALABLE_IMAGE_DATASETS:
        raise ValueError(
            f"Dataset {dataset!r} only supports image_size={default_size}; "
            f"got image_size={int(image_size)}."
        )

    profile_key = _profile_key_for_size(dataset, image_size)
    return profile_key, _sized_dataset_defaults(dataset, image_size), model


def _validate_positive_count(name, value):
    """Require a positive integer count."""
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _validate_split_count(config, *, field_name, split_name, count):
    """Validate a requested subset count when the split size is known."""
    dataset_name = str(config.data.dataset)
    split_sizes = _KNOWN_SPLIT_SIZES.get(dataset_name)
    if split_sizes is None:
        return

    split_key = str(split_name)
    if split_key == "val":
        split_key = "valid"
    split_size = split_sizes.get(split_key)
    if split_size is None:
        return
    if int(count) > int(split_size):
        raise ValueError(
            f"{field_name}={int(count)} exceeds {dataset_name} split "
            f"{split_name!r} size {int(split_size)}"
        )


def _validate_image_counts(config):
    """
    Validate image subset and run counts before dataset construction.

    This catches impossible train/reference sizes before torchvision or local
    folder loaders begin materializing images.
    """
    _validate_positive_count("n_train", config.data.n_train)
    _validate_positive_count("n_test", config.data.n_test)
    _validate_positive_count("eval_sample_bank_size", config.evaluation.n_samples)
    _validate_positive_count("n_training_steps", config.training.n_training_steps)
    _validate_positive_count("ckpt_every", config.training.ckpt_every)
    _validate_positive_count("batch_cap", config.training.batch_cap)
    _validate_positive_count("sampling_steps", config.sampling.n_steps)
    _validate_positive_count(
        "generation_monitor_samples",
        config.generation_monitor.n_samples,
    )
    monitor = config.random_projected_f_mem
    _validate_positive_count("random_projected_f_mem_samples", monitor.n_samples)
    _validate_positive_count("random_projected_f_mem_resize", monitor.resize_size)
    _validate_positive_count("random_projected_f_mem_proj_dim", monitor.projection_dim)

    _validate_split_count(
        config,
        field_name="n_train",
        split_name=getattr(config.data, "train_split", "train"),
        count=config.data.n_train,
    )
    _validate_split_count(
        config,
        field_name="n_test",
        split_name=getattr(config.data, "reference_split", "test"),
        count=config.data.n_test,
    )


def build_config(settings):
    """
    Build the canonical image UNet config from merged run settings.

    Precedence is: base defaults, dataset/size profile, then user-provided
    sweep values. Dataset-owned fields such as splits stay with the profile
    unless a caller explicitly overrides them outside the normal CLI path.
    """
    if isinstance(settings, argparse.Namespace):
        raw_settings = vars(settings)
    else:
        raw_settings = dict(settings)
    dataset, dataset_defaults, model = _resolve_dataset_profile(raw_settings)
    settings = _merge_settings(raw_settings, dataset_defaults)
    settings["dataset"] = dataset
    settings["model"] = model

    def data_setting(key):
        if _raw_overrides_default(raw_settings, key):
            return raw_settings[key]
        return settings[key]

    def model_setting(key):
        if key not in _MODEL_SETTING_KEYS:
            raise KeyError(key)
        if _raw_overrides_default(raw_settings, key):
            return raw_settings[key]
        return settings[key]

    config = clone_config(get_base_config())

    # Study metadata.
    config.study.name = RUNNER_NAME
    config.study.description = f"{dataset} image GDFM/UNet runner."
    config.study.dataset_key = dataset
    config.outputs.run_name = str(settings["run_name"])

    # Framework.
    config.framework.family = "fm"
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
    config.data.n_test = int(settings["n_test"])
    config.data.recipe = str(dataset_defaults.get("recipe", "default"))
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
    config.network.arch = model
    config.network.base_channels = int(model_setting("base_channels"))
    config.network.channel_mult = tuple(model_setting("channel_mult"))
    config.network.attn_resolutions = tuple(model_setting("attn_resolutions"))
    config.network.num_res_blocks = int(model_setting("num_res_blocks"))
    config.network.dropout = float(model_setting("dropout"))
    config.network.time_emb_dim = model_setting("time_emb_dim")

    # Training.
    config.training.seed = int(settings["seed"])
    config.training.n_training_steps = int(settings["n_training_steps"])
    config.training.ckpt_every = int(settings["ckpt_every"])
    config.training.ckpt_schedule = str(settings["ckpt_schedule"])
    config.training.ckpt_steps = parse_checkpoint_steps(settings["ckpt_steps"])
    config.training.resolved_ckpt_steps = tuple(resolve_checkpoint_steps(config))
    log_every = settings.get("log_every")
    if log_every is None:
        config.training.log_every = max(1, min(500, int(config.training.ckpt_every)))
    else:
        config.training.log_every = max(1, int(log_every))
    batch_cap = resolve_batch_cap(settings)
    config.training.batch_cap = batch_cap
    config.training.batch_size = batch_cap
    config.training.optimizer = DEFAULT_IMAGE_OPTIMIZER
    lr = raw_settings["lr"] if _raw_overrides_default(raw_settings, "lr") else None
    config.training.lr = float(_coalesce(lr, DEFAULT_IMAGE_LR))
    weight_decay = (
        raw_settings["weight_decay"]
        if _raw_overrides_default(raw_settings, "weight_decay")
        else None
    )
    config.training.weight_decay = float(
        _coalesce(weight_decay, DEFAULT_IMAGE_WEIGHT_DECAY)
    )

    # Runtime.
    config.runtime.precision = str(settings["precision"])
    config.runtime.tf32_enabled = False
    config.runtime.autocast_dtype = "none"
    config.runtime.pin_memory = True
    config.runtime.non_blocking_h2d = False

    # Evaluation.
    eval_sample_bank_size = int(settings["eval_sample_bank_size"])
    monitor_samples = int(settings["generation_monitor_samples"])
    config.evaluation.sample_bank_size = eval_sample_bank_size
    config.evaluation.n_samples = eval_sample_bank_size
    config.evaluation.batch_size = batch_cap
    config.evaluation.sample_panel_count = min(
        8,
        int(config.evaluation.n_samples),
    )
    loss_eval_times = (
        raw_settings["loss_eval_times"]
        if _raw_overrides_default(raw_settings, "loss_eval_times")
        else DEFAULT_LOSS_EVAL_TIMES
    )
    config.evaluation.loss_eval_times = tuple(float(item) for item in loss_eval_times)
    config.evaluation.loss_eval_batch_size = 0
    config.evaluation.chunk_size = max(
        128,
        int(config.evaluation.n_samples),
        int(config.data.n_test),
        int(config.data.n_train),
    )

    # Sampling.
    config.sampling.method = "heun"
    config.sampling.n_steps = int(settings["sampling_steps"])

    # Generation monitor.
    config.generation_monitor.enabled = bool(settings["generation_monitor"])
    config.generation_monitor.n_samples = monitor_samples
    config.generation_monitor.show = bool(settings["generation_monitor_show"])

    # Random projected f_mem monitor.
    monitor = config.random_projected_f_mem
    monitor.n_samples = int(settings["random_projected_f_mem_samples"])
    monitor.resize_size = int(settings["random_projected_f_mem_resize"])
    monitor.projection_dim = int(settings["random_projected_f_mem_proj_dim"])
    projection_seed = settings.get("random_projected_f_mem_proj_seed")
    if projection_seed is None:
        projection_seed = int(config.training.seed) + 3001
    monitor.projection_seed = int(projection_seed)

    # Tracking.
    config.tracking.console.summary_every_steps = int(config.training.log_every)
    config.tracking.console.summary_every_seconds = float(
        DEFAULT_CONSOLE_SUMMARY_SECONDS
    )
    config.tracking.console.compact_model_summary = bool(settings["compact_model_summary"])
    config.tracking.wandb.enabled = bool(settings["wandb"])
    config.tracking.wandb.mode = str(settings["wandb_mode"])
    if bool(settings.get("wandb_project_provided", False)):
        config.tracking.wandb.project = str(settings["wandb_project"])
    else:
        config.tracking.wandb.project = _resolve_wandb_project(
            settings["wandb_project"],
            dataset,
            config.data.image_size,
            config.network.arch,
        )
    config.tracking.wandb.project_explicit = bool(
        settings.get("wandb_project_provided", False)
    )
    config.tracking.wandb.name_explicit = bool(
        settings.get("wandb_name_provided", False)
    )
    config.tracking.wandb.group_explicit = bool(
        settings.get("wandb_group_provided", False)
    )
    wandb_entity = (
        raw_settings["wandb_entity"]
        if _raw_overrides_default(raw_settings, "wandb_entity")
        else DEFAULT_SETTINGS["wandb_entity"]
    )
    wandb_notes = (
        raw_settings["wandb_notes"]
        if _raw_overrides_default(raw_settings, "wandb_notes")
        else DEFAULT_SETTINGS["wandb_notes"]
    )
    wandb_strict = (
        raw_settings["wandb_strict"]
        if _raw_overrides_default(raw_settings, "wandb_strict")
        else DEFAULT_SETTINGS["wandb_strict"]
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
    resolve_wandb_identity(config)

    _validate_image_counts(config)
    return config


if __name__ == "__main__":
    print("mp_runners.image_config example")
    print("datasets:", ", ".join(list_datasets()))
