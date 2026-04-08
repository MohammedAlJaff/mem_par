import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# Config
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

    # --- time sampling ---
    time_sampling: str = "uniform"  # currently only used for FM
    time_sampling_mu: float = 0.0
    time_sampling_sigma: float = 1.0

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
        if self.family == "fm" and self.time_sampling not in {"uniform"}:
            raise ValueError(
                f"For now, FM time_sampling must be 'uniform', got {self.time_sampling}"
            )
        if self.beta_min <= 0 or self.beta_max <= 0:
            raise ValueError("beta_min and beta_max must be positive")
        if self.beta_max < self.beta_min:
            raise ValueError("beta_max must be >= beta_min")
        if self.time_sampling not in {"uniform", "logit_normal"}:
            raise ValueError(
                f"time_sampling must be 'uniform' or 'logit_normal', got {self.time_sampling}"
            )
        if self.family == "fm":
            if self.fm_time_sampling not in {"uniform", "logit_normal"}:
                raise ValueError(
                    f"fm_time_sampling must be 'uniform' or 'logit_normal', got {self.fm_time_sampling}"
                )
            if self.fm_logit_sigma <= 0:
                raise ValueError("fm_logit_sigma must be positive")


# ============================================================
# Time-conditioned MLP
# ============================================================


class TimeCondMLP(nn.Module):
    """
    Flat-vector MLP with sinusoidal time embedding.
    Input:
        z: [B, D]
        t: [B]
    Output:
        y: [B, D]
    """

    def __init__(
        self, data_dim: int, hidden_dim: int, n_layers: int, time_emb_dim: int
    ):
        super().__init__()
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_emb_dim),
        )

        layers = [nn.Linear(data_dim + time_emb_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, data_dim)]
        self.net = nn.Sequential(*layers)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal embedding of scalar t in [0,1].
        t: [B]
        returns: [B, time_emb_dim]
        """
        half = self.time_emb_dim // 2
        idx = torch.arange(half, device=t.device, dtype=t.dtype)
        freqs = torch.exp(-math.log(10000.0) * idx / max(half - 1, 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.time_mlp(emb)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(
                f"TimeCondMLP expects z shape [B, D], got {tuple(z.shape)}"
            )
        if t.dim() != 1:
            raise ValueError(f"TimeCondMLP expects t shape [B], got {tuple(t.shape)}")
        if z.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch mismatch: z has {z.shape[0]} samples, t has {t.shape[0]} samples"
            )

        t_emb = self.time_embedding(t)
        return self.net(torch.cat([z, t_emb], dim=-1))


# ============================================================
# GDFM
# ============================================================


class GDFM(nn.Module):
    """
    Generalized linear-target framework for:
        - Flow Matching (FM)
        - VP / DDPM-style perturbation paths

    Convention:
        t = 0 -> noise / source end
        t = 1 -> data end

    Core objects:
        z_t = a(t) x + b(t) eps
        r_t(space) = c_space(t) x + d_space(t) eps
    """

    def __init__(self, config: GDFMConfig):
        super().__init__()
        self.config = config

        self._setup_family(config)
        self._setup_network(config)

    # --------------------------------------------------------
    # Family setup
    # --------------------------------------------------------

    def _setup_family(self, cfg: GDFMConfig) -> None:
        """
        Assign family-specific coefficient functions and time sampler.

        After this:
            self.a(t), self.b(t), self.w(t), self._sample_t(B, device)
        are all defined.
        """
        self.family_name = cfg.family
        self.v_kind = cfg.family

        if cfg.family == "fm":
            self.a = self._a_fm
            self.b = self._b_fm
            self.w = self._w_const

            if cfg.fm_time_sampling == "uniform":
                self._sample_t = self._sample_t_uniform
            elif cfg.fm_time_sampling == "logit_normal":
                self._sample_t = self._sample_t_logit_normal
            else:
                raise ValueError(f"Unknown fm_time_sampling: {cfg.fm_time_sampling}")

        elif cfg.family == "vp":
            self.a = self._a_vp
            self.b = self._b_vp
            self.w = self._w_const
            self._sample_t = self._sample_t_uniform

        else:
            raise ValueError(f"Unknown family: {cfg.family}")

    # --------------------------------------------------------
    # Family-specific time sampling p(t)
    # --------------------------------------------------------

    def _sample_t_uniform(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Uniform time sampling:
            t ~ Uniform(t_min, t_max)
        """
        cfg = self.config
        return (
            torch.rand(batch_size, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min
        )

    def _sample_t_logit_normal(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        FM logit-normal time sampling.

        s ~ N(mu, sigma^2)
        t = sigmoid(s)

        Concentrates mass near endpoints depending on (mu, sigma).
        """
        cfg = self.config
        s = cfg.fm_logit_mu + cfg.fm_logit_sigma * torch.randn(
            batch_size, device=device
        )
        t = torch.sigmoid(s)
        return t.clamp(cfg.t_min, cfg.t_max)

    # --------------------------------------------------------
    # FM coefficients
    # --------------------------------------------------------

    def _a_fm(self, t: torch.Tensor) -> torch.Tensor:
        """
        FM signal coefficient:
            a(t) = t
        """
        return t

    def _b_fm(self, t: torch.Tensor) -> torch.Tensor:
        """
        FM noise coefficient:
            b(t) = 1 - t
        """
        return 1.0 - t

    # --------------------------------------------------------
    # VP coefficients
    # --------------------------------------------------------

    def _a_vp(self, t: torch.Tensor) -> torch.Tensor:
        """
        VP signal coefficient:
            a(t) = sqrt(alpha_bar(s)), with s = 1 - t

        alpha_bar(s) = exp(- ∫_0^s beta(u) du)
        """
        s = 1.0 - t
        bmin, bmax = self.config.beta_min, self.config.beta_max
        int_beta = bmin * s + 0.5 * (bmax - bmin) * s**2
        return torch.exp(-0.5 * int_beta)

    def _b_vp(self, t: torch.Tensor) -> torch.Tensor:
        """
        VP noise coefficient:
            b(t) = sqrt(1 - a(t)^2)
        """
        a = self._a_vp(t)
        return torch.sqrt(torch.clamp(1.0 - a**2, min=0.0))

    # --------------------------------------------------------
    # Weighting w(t)
    # --------------------------------------------------------

    def _w_const(self, t: torch.Tensor) -> torch.Tensor:
        """
        Constant weighting:
            w(t) = 1
        """
        return torch.ones_like(t)

    # --------------------------------------------------------
    # Network setup
    # --------------------------------------------------------

    def _setup_network(self, cfg: GDFMConfig) -> None:
        if cfg.network == "mlp":
            self.net = TimeCondMLP(
                data_dim=cfg.data_dim,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.n_layers,
                time_emb_dim=cfg.time_emb_dim,
            )
        else:
            raise ValueError(f"Unknown network: {cfg.network}")

    # --------------------------------------------------------
    # Time and path
    # --------------------------------------------------------

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self._sample_t(batch_size, device)

    def interpolate(
        self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        z_t = a(t) x + b(t) eps
        """
        t_b = self._bcast(t, x)
        return self.a(t_b) * x + self.b(t_b) * eps

    # --------------------------------------------------------
    # Target coefficients
    # --------------------------------------------------------

    def target_coeffs(
        self, space: str, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns coefficients (c(t), d(t)) such that

            r_t(space) = c(t) x + d(t) eps
        """
        if space == "x":
            c = torch.ones_like(t)
            d = torch.zeros_like(t)

        elif space == "eps":
            c = torch.zeros_like(t)
            d = torch.ones_like(t)

        elif space == "v":
            if self.v_kind == "fm":
                c = torch.ones_like(t)
                d = -torch.ones_like(t)
            elif self.v_kind == "vp":
                c = -self.b(t)
                d = self.a(t)
            else:
                raise ValueError(f"Unknown v_kind: {self.v_kind}")

        else:
            raise ValueError(f"Unknown space: {space}")

        return c, d

    def get_target(
        self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, space: str
    ) -> torch.Tensor:
        """
        r_t(space) = c(t) x + d(t) eps
        """
        t_b = self._bcast(t, x)
        c, d = self.target_coeffs(space, t_b)
        return c * x + d * eps

    # --------------------------------------------------------
    # Prediction and conversion
    # --------------------------------------------------------

    def predict(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(z, t)

    def convert_prediction(
        self,
        y_pred: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        from_space: str,
        to_space: str,
    ) -> torch.Tensor:
        """
        Convert a predicted target from `from_space` to `to_space`.
        """
        if from_space == to_space:
            return y_pred

        t_b = self._bcast(t, z)

        a = self.a(t_b)
        b = self.b(t_b)
        c_from, d_from = self.target_coeffs(from_space, t_b)
        c_to, d_to = self.target_coeffs(to_space, t_b)

        det = a * d_from - b * c_from

        det_abs_min = det.abs().amin().item()
        if det_abs_min < 1e-8:
            raise RuntimeError(
                f"Conversion became nearly singular for from_space={from_space}. "
                f"Smallest |det| = {det_abs_min:.3e}. "
                f"Try tighter endpoint clipping."
            )

        x_hat = (d_from * z - b * y_pred) / det
        eps_hat = (-c_from * z + a * y_pred) / det

        return c_to * x_hat + d_to * eps_hat

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    def compute_loss(
        self,
        x: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full training step:
            eps ~ N(0, I)
            t   ~ p(t)
            z   = a(t) x + b(t) eps
        """
        if x.dim() != 2:
            raise ValueError(
                f"Current implementation expects flat x with shape [B, D], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.config.data_dim:
            raise ValueError(
                f"x.shape[1]={x.shape[1]} but config.data_dim={self.config.data_dim}"
            )

        B = x.shape[0]
        device = x.device

        if eps is None:
            eps = torch.randn_like(x)
        if t is None:
            t = self.sample_t(B, device)

        if eps.shape != x.shape:
            raise ValueError(
                f"eps shape must match x shape, got {tuple(eps.shape)} vs {tuple(x.shape)}"
            )
        if t.shape != (B,):
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")

        z = self.interpolate(x, eps, t)
        y_pred = self.predict(z, t)
        y_loss = self.convert_prediction(
            y_pred=y_pred,
            z=z,
            t=t,
            from_space=self.config.prediction_space,
            to_space=self.config.loss_space,
        )
        r_loss = self.get_target(x, eps, t, self.config.loss_space)

        t_b = self._bcast(t, x)
        w = self.w(t_b)

        per_sample = (w * (y_loss - r_loss) ** 2).flatten(1).mean(1)
        return per_sample.mean()

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _bcast(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Expand t: [B] -> [B,1,...,1] to match ref shape [B,...]
        """
        if t.dim() != 1:
            raise ValueError(f"Expected t to have shape [B], got {tuple(t.shape)}")
        if ref.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch mismatch: ref has B={ref.shape[0]}, t has B={t.shape[0]}"
            )
        return t.view([t.shape[0]] + [1] * (ref.dim() - 1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            "GDFM(\n"
            f"  family           = {self.family_name}\n"
            f"  prediction_space = {self.config.prediction_space}\n"
            f"  loss_space       = {self.config.loss_space}\n"
            f"  v_kind           = {self.v_kind}\n"
            f"  params           = {self.n_params():,}\n"
            f"  network          = {self.config.network}\n"
            ")"
        )
