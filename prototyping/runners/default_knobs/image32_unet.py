"""Default knob surfaces for `prototyping.runners.image32_unet`.

Keeping these values in their own module lets us reuse consistent defaults
across the CLI, notebooks, and batch scripts while making dataset-specific
recipes explicit.
"""

import os

from prototyping.runners.metrics_image32 import DEFAULT_FIXED_TIMES


DEFAULT_STUDY_DIR = "outputs/prototype_runners/image32_unet"


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


DATASET_KNOB_RECIPES = {
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


def _default_wandb_project(fallback):
    project = os.getenv("WANDB_PROJECT", "").strip()
    return project if project else fallback


def default_knobs():
    """Return the editable default knobs for the Image32 UNet runner."""
    return {
        "device": "cpu",
        "out_dir": DEFAULT_STUDY_DIR,
        "run_name": "image32_unet",
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


def dataset_knob_recipes():
    """Return a copy of all exported dataset-specific defaults."""
    return {name: dict(values) for name, values in DATASET_KNOB_RECIPES.items()}
