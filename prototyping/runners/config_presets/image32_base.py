"""Base Image32 UNet config defaults (shared across presets)."""

import os
from ml_collections import ConfigDict


def _default_wandb_project(fallback):
    return os.getenv("WANDB_PROJECT", "").strip() or fallback


def _default_wandb_entity(fallback):
    return os.getenv("WANDB_ENTITY", "").strip() or fallback


def get_default_configs():
    """Return shared Image32 UNet defaults as an `ml_collections.ConfigDict`."""
    config = ConfigDict()

    # Study + framework.
    config.study = ConfigDict()
    config.study.name = "image_runner"
    config.study.description = "Image UNet prototype recipe."

    config.framework = ConfigDict()
    config.framework.family = "fm"
    config.framework.prediction_space = "v"
    config.framework.loss_space = "v"

    # Model defaults.
    config.model = ConfigDict()
    config.model.arch = "unet"
    config.model.base_channels = 32
    config.model.channel_mult = (1, 2, 4)
    config.model.attn_resolutions = (16, 8)
    config.model.num_res_blocks = 2
    config.model.dropout = 0.1
    config.model.time_emb_dim = None

    # Preset-owned image geometry. Dataset profiles still own channel count,
    # splits, and source directories.
    config.data = ConfigDict()
    config.data.image_size = 32
    config.data.resize_size = 0

    # Training defaults.
    config.training = ConfigDict()
    config.training.seed = 0
    config.training.n_steps = 30000
    config.training.ckpt_every = 2000
    config.training.log_every = 200
    config.training.optimizer = "adam"
    config.training.lr = 1.0e-4
    config.training.weight_decay = 0.0

    # Evaluation / sampling defaults.
    config.evaluation = ConfigDict()
    config.evaluation.n_samples = 64
    config.evaluation.sample_panel_count = 8
    config.evaluation.chunk_size = 128
    config.evaluation.loss_eval_times = (0.5, 0.75)
    config.evaluation.loss_eval_batch_size = 0

    config.sampling = ConfigDict()
    config.sampling.method = "heun"
    config.sampling.n_steps = 24

    # Tracking defaults.
    config.tracking = ConfigDict()
    config.tracking.wandb = ConfigDict()
    config.tracking.wandb.enabled = True
    config.tracking.wandb.mode = "online"
    config.tracking.wandb.project = _default_wandb_project("")
    config.tracking.wandb.entity = _default_wandb_entity("maj90")
    config.tracking.wandb.notes = ""
    config.tracking.wandb.name = ""
    config.tracking.wandb.group = ""
    config.tracking.wandb.tags = []
    config.tracking.wandb.strict = False

    config.tracking.console = ConfigDict()
    config.tracking.console.summary_every_seconds = 60.0

    return config
