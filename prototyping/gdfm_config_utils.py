from ml_collections import ConfigDict
from dataclasses import dataclass

# ============================================================
# Config: ConfigDict version.
# ============================================================


def get_default_config():
    config = ConfigDict()

    # ---- framework ----
    framework = config.framework = ConfigDict()
    framework.family = "fm"  # "fm" or "vp"
    framework.prediction_space = "x"  # "x", "eps", "v"
    framework.loss_space = "v"  # "x", "eps", "v"

    # ---- schedule ----
    schedule = config.schedule = ConfigDict()
    schedule.t_min = 1e-3
    schedule.t_max = 0.999
    schedule.denom_eps = 0.05  # Li & He (2025) clamp floor
    # VP-specific
    schedule.beta_min = 0.1
    schedule.beta_max = 20.0
    # FM time sampling
    schedule.time_sampling = "uniform"  # "uniform" or "logit_normal"
    schedule.logit_mu = 0.0
    schedule.logit_sigma = 1.0

    # ---- network ----
    network = config.network = ConfigDict()
    network.arch = "mlp"  # "mlp", "unet", "jit"
    # MLP
    network.hidden_dim = 128
    network.n_layers = 3
    network.time_emb_dim = 32
    # UNet (used when arch = "unet")
    network.base_channels = 64
    network.channel_mult = (1, 2, 4, 4)
    network.attn_resolutions = (16,)
    network.num_res_blocks = 2
    network.dropout = 0.0
    # JiT (used when arch = "jit")
    network.patch_size = 16
    network.depth = 12
    network.heads = 12
    network.bottleneck_dim = 128
    network.proj_dropout = 0.0

    # ---- data ----
    data = config.data = ConfigDict()
    data.dataset = "ring"  # "ring", "moons", "gaussians",
    # "celeba32_grey", "celeba64_rgb",
    # "afhq64_rgb", "torus", "gmm_subspace"
    data.data_dim = 2  # ambient dimension D
    data.intrinsic_dim = 2  # intrinsic dimension d (known for synthetic)
    data.image_size = 0  # H=W for image datasets, 0 for flat
    data.image_channels = 0  # C for image datasets, 0 for flat
    data.n_train = 2000  # training set size (subsampled)
    data.n_eval = 2000  # held-out reference set size
    data.noise = 0.05  # observation noise (synthetic only)
    # GMM subspace specific
    data.n_clusters = 10
    data.sigma_z = 1.0
    data.sigma_eta = 0.01
    # torus specific
    data.torus_dim = 2  # d for torus embedding

    # ---- sampling ----
    sampling = config.sampling = ConfigDict()
    sampling.method = "euler"  # "euler" or "heun"
    sampling.n_steps = 200
    sampling.n_samples = 2000

    # ---- training ----
    training = config.training = ConfigDict()
    training.n_steps = 5000
    training.batch_cap = 256
    training.batch_size = 256
    training.lr = 3e-4
    training.weight_decay = 1e-4
    training.warmup_frac = 0.05  # fraction of training for LR warmup
    training.ema_decay = 0.9999
    training.seed = 42
    training.ckpt_every = 500  # checkpoint interval (steps)
    training.log_every = 100  # logging interval (steps)
    training.n_seeds = 3  # number of random seeds per config

    return config


# ============================================================
# Config: Data class version.
# ============================================================


@dataclass
class GDFMConfig:
    """
    usage: config = GDFMConfig(family="vp", prediction_space="eps", loss_space="eps")
    """

    # --- family / objective ---
    family: str  # "fm" or "vp"
    prediction_space: str  # "x", "eps", "v"
    loss_space: str  # "x", "eps", "v"

    # --- network ---
    data_dim: int = 784
    network: str = "mlp"
    hidden_dim: int = 256
    n_layers: int = 3
    time_emb_dim: int = 64

    # --- time domain ---
    t_min: float = 1e-3
    t_max: float = 1.0 - 1e-3

    # --- safe division ---
    safe_div_eps: float = 0.05  # Li & He (2025) use 0.05

    # --- FM time sampling ---
    fm_time_sampling: str = "uniform"  # "uniform" or "logit_normal"
    fm_logit_mu: float = 0.0
    fm_logit_sigma: float = 1.0

    # --- VP schedule ---
    beta_min: float = 0.1
    beta_max: float = 20.0

    def __post_init__(self):
        if self.family not in {"fm", "vp"}:
            raise ValueError(f"family must be 'fm' or 'vp', got {self.family}")

        if self.prediction_space not in {"x", "eps", "v"}:
            raise ValueError(
                f"prediction_space must be one of {{'x','eps','v'}}, got {self.prediction_space}"
            )

        if self.loss_space not in {"x", "eps", "v"}:
            raise ValueError(
                f"loss_space must be one of {{'x','eps','v'}}, got {self.loss_space}"
            )

        if self.network != "mlp":
            raise ValueError(
                f"Only network='mlp' is implemented here, got {self.network}"
            )

        if self.time_emb_dim % 2 != 0:
            raise ValueError(f"time_emb_dim must be even, got {self.time_emb_dim}")

        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")

        if self.data_dim < 1:
            raise ValueError(f"data_dim must be >= 1, got {self.data_dim}")

        if not (0.0 <= self.t_min < self.t_max <= 1.0):
            raise ValueError(
                f"Require 0 <= t_min < t_max <= 1, got ({self.t_min}, {self.t_max})"
            )

        if self.beta_min <= 0 or self.beta_max <= 0:
            raise ValueError("beta_min and beta_max must be positive")

        if self.beta_max < self.beta_min:
            raise ValueError("beta_max must be >= beta_min")

        if self.family == "fm":
            if self.fm_time_sampling not in {"uniform", "logit_normal"}:
                raise ValueError(
                    f"fm_time_sampling must be 'uniform' or 'logit_normal', got {self.fm_time_sampling}"
                )
            if self.fm_time_sampling == "logit_normal" and self.fm_logit_sigma <= 0:
                raise ValueError("fm_logit_sigma must be positive")
