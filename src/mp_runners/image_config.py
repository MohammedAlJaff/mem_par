"""
Configuration builder for the image UNet runner.

The runner receives simple CLI/settings values. This module turns them into the
canonical nested config used by data loading, model construction, training,
evaluation, runtime precision, and tracking.
"""

import copy
import os
import re

from mp_config import clone_config, get_base_config
from mp_metrics.loss_eval import DEFAULT_LOSS_EVAL_TIMES
from mp_runners.common import resolve_batch_cap
from mp_runners.image_presets import get_default_configs
from mp_training.checkpoint_schedule import (
    parse_checkpoint_steps,
    resolve_checkpoint_steps,
)
from mp_tracking import resolve_wandb_identity


RUNNER_NAME = "image_runner"
DEFAULT_STUDY_DIR = "outputs/runners/image_runner"
MODEL_CHOICES = ("unet",)


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


def _dataset_model_wandb_project(dataset_key, model_arch):
    """Build the default W&B project name for a dataset/model pair."""
    dataset = _wandb_slug(dataset_key, "dataset")
    model = _wandb_slug(model_arch, "model")
    return f"memparam-{dataset}-{model}"


def _resolve_wandb_project(raw_project, dataset_key, model_arch):
    """Resolve explicit, legacy, and automatic W&B project names."""
    requested = str(raw_project or "").strip()
    canonical = _dataset_model_wandb_project(dataset_key, model_arch)
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
        # Flowers102's official train split has only 1,020 images.
        # For generative runs we train on the large official test split and use
        # the small official train split as the held-out/reference set.
        "train_split": "test",
        "reference_split": "train",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[32],
    },
    "flowers102_64": {
        "dataset": "flowers102",
        "config_profile": "image32_64",
        "image_size": 64,
        "image_channels": 3,
        "resize_size": 64,
        # Flowers102's official train split has only 1,020 images.
        # For generative runs we train on the large official test split and use
        # the small official train split as the held-out/reference set.
        "train_split": "test",
        "reference_split": "train",
        "allow_same_source_reference": False,
        "dataset_subdir": "",
        **_UNET_PROFILES[64],
    },
    "flowers102_128": {
        "dataset": "flowers102",
        "config_profile": "image32_128",
        "image_size": 128,
        "image_channels": 3,
        "resize_size": 128,
        # Flowers102's official train split has only 1,020 images.
        # For generative runs we train on the large official test split and use
        # the small official train split as the held-out/reference set.
        "train_split": "test",
        "reference_split": "train",
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
        # Flowers102's official train split has only 1,020 images.
        # For generative runs we train on the large official test split and use
        # the small official train split as the held-out/reference set.
        "train_split": "test",
        "reference_split": "train",
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


SCALABLE_IMAGE_DATASETS = {"mnist", "flowers102", "afhq_v2"}
DATASET_ALIASES = {
    "ffhq": "ffhq128",
}


DEFAULT_SETTINGS = {
    # Run identity and reproducibility.
    "device": "cpu",
    "out_dir": DEFAULT_STUDY_DIR,
    "run_name": RUNNER_NAME,
    "seed": 0,

    # Data/model runner selection.
    "dataset": "celeba32_grey",
    "config_preset": None,
    "model": "unet",
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
    "ckpt_schedule": "bonnaire-1m",
    "ckpt_steps": "",
    "log_every": None,
    "batch_cap": 64,

    # Runtime numerics.
    # precision: fp32 preserves current behavior; tf32/bf16 are CUDA speed paths.
    "precision": "fp32",
    "compare_numerics": False,
    "data_summary": False,
    "final_eval": False,
    "eval_run_dir": None,

    # Sampling/evaluation.
    # eval_samples: generated samples per checkpoint; also the size of z_init.pt.
    # sampling_steps: numerical solver steps used when turning z_init into images.
    "eval_samples": 64,
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
    return tuple(sorted(set(DATASET_DEFAULTS.keys()) | set(DATASET_ALIASES.keys())))


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
            key in _PRESET_OWNED_SETTING_KEYS
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
    """Return the user-facing dataset key for a base dataset and image size."""
    candidate = f"{dataset}_{int(image_size)}"
    if candidate in DATASET_DEFAULTS:
        return candidate
    default_size = int(DATASET_DEFAULTS[dataset]["image_size"])
    if int(image_size) == default_size:
        return dataset
    return candidate


def _sized_dataset_defaults(dataset, image_size):
    """Build dataset defaults for a supported base dataset at a requested size."""
    image_size = int(image_size)
    defaults = copy.deepcopy(DATASET_DEFAULTS[dataset])
    if image_size in _UNET_PROFILES:
        defaults.update(_UNET_PROFILES[image_size])
        defaults["config_profile"] = f"image32_{image_size}"
    defaults["image_size"] = image_size
    defaults["resize_size"] = 0 if image_size == 32 else image_size
    return defaults


def _resolve_dataset_profile(raw_settings):
    """
    Resolve dataset, image size, and model into one concrete dataset profile.

    This is the main convenience path for the image runner: callers can pass a
    base dataset such as `flowers102`, an image size such as `64`, and a model
    such as `unet`; the runner chooses the matching data/model defaults.
    """
    model = _resolve_model(raw_settings.get("model", DEFAULT_SETTINGS["model"]))
    raw_dataset = str(_coalesce(raw_settings.get("dataset"), DEFAULT_SETTINGS["dataset"]))
    dataset = DATASET_ALIASES.get(raw_dataset, raw_dataset)
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown image runner dataset: {raw_dataset}")

    image_size = _resolve_requested_image_size(raw_settings.get("image_size"))
    defaults = DATASET_DEFAULTS[dataset]
    if image_size is None:
        return dataset, defaults, model

    default_size = int(defaults["image_size"])
    if int(image_size) == default_size:
        return dataset, defaults, model

    if dataset != str(defaults["dataset"]):
        raise ValueError(
            f"Dataset key {dataset!r} already fixes image_size={default_size}. "
            f"Use --dataset {defaults['dataset']} --image-size {int(image_size)}."
        )

    if dataset not in SCALABLE_IMAGE_DATASETS:
        raise ValueError(
            f"Dataset {dataset!r} does not support automatic image_size={int(image_size)}."
        )

    profile_key = _profile_key_for_size(dataset, image_size)
    if profile_key in DATASET_DEFAULTS:
        return profile_key, DATASET_DEFAULTS[profile_key], model
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
    _validate_positive_count("n_eval", config.data.n_eval)
    _validate_positive_count("eval_samples", config.evaluation.n_samples)
    _validate_positive_count("n_steps", config.training.n_steps)
    _validate_positive_count("ckpt_every", config.training.ckpt_every)
    _validate_positive_count("batch_cap", config.training.batch_cap)
    _validate_positive_count("sampling_steps", config.sampling.n_steps)
    _validate_positive_count("generation_trace_count", config.generation_trace.num_images)

    _validate_split_count(
        config,
        field_name="n_train",
        split_name=getattr(config.data, "train_split", "train"),
        count=config.data.n_train,
    )
    _validate_split_count(
        config,
        field_name="n_eval",
        split_name=getattr(config.data, "reference_split", "test"),
        count=config.data.n_eval,
    )


def _preset_value(defaults, section_name, key, fallback):
    """Read a value from a preset section with a fallback."""
    section = getattr(defaults, section_name, None)
    if section is None:
        return fallback
    try:
        value = getattr(section, key)
    except AttributeError:
        return fallback
    return _coalesce(value, fallback)


def build_config(settings):
    """
    Build the canonical image UNet config from merged run settings.

    Precedence is: base defaults, dataset profile, optional explicit preset,
    then user-provided sweep values. Dataset-owned fields such as image size and
    split choice stay with the profile unless a caller explicitly overrides
    them outside the normal CLI path.
    """
    raw_settings = dict(settings)
    dataset, dataset_defaults, model = _resolve_dataset_profile(raw_settings)
    settings = _merge_settings(raw_settings, dataset_defaults)
    settings["dataset"] = dataset
    settings["model"] = model

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
    config.study.description = f"{dataset} image GDFM/UNet runner."
    config.study.dataset_key = dataset
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
    config.network.arch = model
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

    # Runtime.
    config.runtime.precision = str(settings["precision"])
    config.runtime.tf32_enabled = False
    config.runtime.autocast_dtype = "none"
    config.runtime.pin_memory = True
    config.runtime.non_blocking_h2d = False

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
    if bool(settings.get("wandb_project_provided", False)):
        config.tracking.wandb.project = str(settings["wandb_project"])
    else:
        config.tracking.wandb.project = _resolve_wandb_project(
            settings["wandb_project"],
            dataset,
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
    resolve_wandb_identity(config)

    _validate_image_counts(config)
    return config


if __name__ == "__main__":
    print("mp_runners.image_config example")
    print("datasets:", ", ".join(list_datasets()))
