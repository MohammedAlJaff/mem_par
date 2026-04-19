"""Configuration builder for the Image32 UNet runner."""

import copy
import os

from prototyping.runners.common import resolve_batch_cap
from prototyping.runners.config_presets import get_default_configs
from mp_config import clone_config, get_base_config
from prototyping.runners.metrics_image32 import DEFAULT_FIXED_TIMES


RUNNER_NAME = "image32_unet"
DEFAULT_STUDY_DIR = "outputs/prototype_runners/image32_unet"


def _default_wandb_project(fallback):
    project = os.getenv("WANDB_PROJECT", "").strip()
    return project if project else fallback


_UNET_PROFILES = {
    32: {
        "base_channels": 64,
        "channel_mult": (1, 2, 4, 4),
        "attn_resolutions": (8,),
        "num_res_blocks": 2,
    },
    128: {
        "base_channels": 64,
        "channel_mult": (1, 2, 4, 4, 8),
        "attn_resolutions": (16,),
        "num_res_blocks": 2,
    },
    256: {
        "base_channels": 64,
        "channel_mult": (1, 2, 4, 4, 8, 8),
        "attn_resolutions": (16,),
        "num_res_blocks": 2,
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
    "device": "cpu",
    "out_dir": DEFAULT_STUDY_DIR,
    "run_name": RUNNER_NAME,
    "seed": 0,
    "dataset": "celeba32_grey",
    "config_preset": None,
    "prediction_space": "v",
    "loss_space": "v",
    "n_train": 128,
    "n_eval": 128,
    "n_steps": 30000,
    "ckpt_every": 2000,
    "batch_cap": 64,
    "eval_samples": 64,
    "sampling_steps": 24,
    "root": "data",
    "dataset_subdir": None,
    "download": False,
    "train_split": "train",
    "reference_split": None,
    "allow_same_source_reference": None,
    "image_size": 32,
    "image_channels": 1,
    "resize_size": 0,
    "balance_train_classes": True,
    "balance_reference_classes": True,
    "loss_eval_times": DEFAULT_FIXED_TIMES,
    "base_channels": 64,
    "channel_mult": (1, 2, 4, 4),
    "attn_resolutions": (8,),
    "num_res_blocks": 2,
    "dropout": 0.0,
    "time_emb_dim": None,
    "wandb": True,
    "wandb_mode": "online",
    "wandb_project": _default_wandb_project("mem_par"),
    "wandb_entity": "",
    "wandb_name": "",
    "wandb_group": "",
    "wandb_tags": "",
    "wandb_notes": "",
    "wandb_strict": False,
    "sanity_updates": 0,
    "sanity_only": False,
}


def get_default_settings():
    """Return canonical base defaults for image32 UNet runs."""
    return copy.deepcopy(DEFAULT_SETTINGS)


def list_datasets():
    """Return supported dataset keys for this runner."""
    return tuple(DATASET_DEFAULTS.keys())


def _coalesce(value, fallback=None):
    if value is None:
        return fallback
    return value


def build_config(settings):
    """Build the canonical Image32 UNet config from merged run settings."""
    settings = dict(settings)
    dataset = str(settings.get("dataset", DEFAULT_SETTINGS["dataset"]))
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown Image32 dataset: {dataset}")
    dataset_defaults = DATASET_DEFAULTS[dataset]
    settings = {**DEFAULT_SETTINGS, **dataset_defaults, **settings}

    preset_name = str(
        settings.get("config_preset", dataset_defaults.get("config_profile", "image32_32"))
    )
    defaults = get_default_configs(preset_name)

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
        settings["dataset_subdir"]
        if settings["dataset_subdir"] is not None
        else dataset_defaults["dataset_subdir"]
    )
    config.data.download = bool(settings["download"])
    config.data.train_split = str(settings["train_split"])
    config.data.reference_split = str(
        settings["reference_split"]
        if settings["reference_split"] is not None
        else dataset_defaults["reference_split"]
    )
    config.data.allow_same_source_reference = bool(
        settings["allow_same_source_reference"]
        if settings["allow_same_source_reference"] is not None
        else dataset_defaults["allow_same_source_reference"]
    )
    config.data.n_train = int(settings["n_train"])
    config.data.n_eval = int(settings["n_eval"])
    config.data.image_size = int(settings["image_size"])
    config.data.image_channels = int(settings["image_channels"])
    config.data.resize_size = int(settings["resize_size"])
    config.data.flatten = False
    config.data.data_dim = (
        int(settings["image_channels"])
        * int(settings["image_size"])
        * int(settings["image_size"])
    )
    config.data.balance_train_classes = bool(settings["balance_train_classes"])
    config.data.balance_reference_classes = bool(settings["balance_reference_classes"])

    # Model / network.
    config.network.arch = defaults.model.arch
    config.network.base_channels = int(settings["base_channels"])
    config.network.channel_mult = tuple(
        _coalesce(settings["channel_mult"], defaults.model.channel_mult)
    )
    config.network.attn_resolutions = tuple(
        _coalesce(settings["attn_resolutions"], defaults.model.attn_resolutions)
    )
    config.network.num_res_blocks = int(settings["num_res_blocks"])
    config.network.dropout = float(settings["dropout"])
    config.network.time_emb_dim = settings["time_emb_dim"]

    # Training.
    config.training.seed = int(settings["seed"])
    config.training.n_steps = int(settings["n_steps"])
    config.training.ckpt_every = int(settings["ckpt_every"])
    config.training.log_every = max(1, min(500, int(config.training.ckpt_every)))
    batch_cap = resolve_batch_cap(settings)
    config.training.batch_cap = batch_cap
    config.training.batch_size = batch_cap
    config.training.optimizer = defaults.training.optimizer
    config.training.lr = float(_coalesce(settings.get("lr"), defaults.training.lr))
    config.training.weight_decay = float(
        _coalesce(settings.get("weight_decay"), defaults.training.weight_decay)
    )

    # Evaluation.
    config.evaluation.n_samples = int(_coalesce(settings["eval_samples"], defaults.evaluation.n_samples))
    config.evaluation.batch_size = batch_cap
    config.evaluation.sample_panel_count = min(
        int(_coalesce(defaults.evaluation.sample_panel_count, 8)),
        int(config.evaluation.n_samples),
    )
    config.evaluation.loss_eval_times = tuple(
        float(item)
        for item in _coalesce(
            settings["loss_eval_times"], defaults.evaluation.loss_eval_times
        )
    )
    config.evaluation.chunk_size = max(
        int(_coalesce(defaults.evaluation.chunk_size, 128)),
        int(config.evaluation.n_samples),
        int(config.data.n_eval),
        int(config.data.n_train),
    )

    # Sampling.
    config.sampling.method = defaults.sampling.method
    config.sampling.n_steps = int(_coalesce(settings["sampling_steps"], defaults.sampling.n_steps))

    # Tracking.
    config.tracking.console.summary_every_steps = int(config.training.log_every)
    config.tracking.console.summary_every_seconds = float(
        defaults.tracking.console.summary_every_seconds
    )
    config.tracking.wandb.enabled = bool(settings["wandb"])
    config.tracking.wandb.mode = str(settings["wandb_mode"])
    config.tracking.wandb.project = str(settings["wandb_project"])
    config.tracking.wandb.entity = str(settings["wandb_entity"])
    config.tracking.wandb.notes = str(settings["wandb_notes"])
    config.tracking.wandb.name = str(settings["wandb_name"])
    config.tracking.wandb.group = str(settings["wandb_group"])
    config.tracking.wandb.tags = [
        item.strip()
        for item in str(settings["wandb_tags"]).split(",")
        if item.strip()
    ]
    config.tracking.wandb.strict = bool(settings["wandb_strict"])
    return config
