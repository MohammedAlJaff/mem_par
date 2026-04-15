import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ============================================================
# GDFM: Generalized Denoising Flow Matching
#
# Time convention:
#   t = 0  →  pure noise (source)
#   t = 1  →  clean data (target)
#   This follows Lipman et al. (2022) and is REVERSED relative
#   to the standard VP/DDPM convention (Song et al. 2021) where
#   t=0 is data and t=T is noise. We handle VP by substituting
#   s = 1 - t internally.
#
# Design:
#   The framework is defined by four scalar functions:
#       a(t), b(t)     — interpolation coefficients
#       a'(t), b'(t)   — their time derivatives (velocity coefficients)
#   Everything else (targets, conversions, loss, sampling) derives
#   from these four functions mechanically. Adding a new family
#   means implementing these four + a time sampler.
#
# The 3×3 grid:
#   prediction_space controls what the network outputs (x, eps, v)
#   loss_space controls where MSE is computed (x, eps, v)
#   These are independent choices. Switching prediction_space is
#   NOT just a loss reweighting — the network's internal
#   representation changes. See Li & He (2025), Table 2.
#
# Velocity definition:
#   v_t = dz_t/dt = a'(t) x + b'(t) eps
#   For FM this gives v = x - eps (constant in t).
#   For VP this gives a schedule-weighted combination that is
#   proportional to (but not equal to) the Salimans & Ho (2022)
#   v-target. See Appendix A for the derivation.
# ============================================================


# ============================================================
# Numerical safety notes
#
# Every conversion ultimately divides by one of three quantities:
#
#   1. b(t)          — vanishes at t=1 (data end)
#   2. a(t)          — vanishes at t=0 (noise end)
#   3. a'b - ab'     — never vanishes (proved below)
#
# Primary defense:  t_min / t_max clipping keeps t away from
#                   endpoints. Training (sample_t) and sampling
#                   (linspace grid) both respect these bounds.
#
# Secondary defense: _safe_div clamps denominators to abs >= 1e-7.
#                    This catches floating point edge cases without
#                    silently producing wrong results far from the
#                    endpoints.
#
# Why (3) is always safe:
#   FM:  a=t, b=1-t, a'=1, b'=-1
#        → a'b - ab' = (1)(1-t) - (t)(-1) = 1           always
#   VP:  a'= beta/2 * a,  b'= -beta*a^2/(2b)
#        → a'b - ab' = beta*a/2 * (b + a^2/b)
#                     = beta*a / (2b)                     > 0
#        since a>0, b>0, beta>0 for t in (0,1)
#
# Which conversions hit which denominator:
#   _x_hat_to_eps_hat   →  divides by b(t)       [case 1]
#   _eps_hat_to_x_hat   →  divides by a(t)       [case 2]
#   _v_hat_to_x_hat     →  divides by a'b - ab'  [case 3]
#   _db_dt_vp            →  divides by b(t)       [case 1]
#
# Everything else (_x_hat_to_v_hat, _eps_hat_to_v_hat,
# _v_hat_to_eps_hat) chains through the above, so no
# additional divisions.
# ============================================================

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


## ============================================================
## Conversion logic
##
## Everything derives from the interpolation and its derivative:
##     z_t = a(t) x + b(t) eps
##     v_t = a'(t) x + b'(t) eps    (ODE velocity, dz_t/dt)
##
## Given any one of (x_hat, eps_hat, v_hat) and z_t, we can
## recover the other two by substitution. All six converters
## route through this same algebra.
## ============================================================


"""
A GDFM is a generalized flow matching model that can be used for:
- Flow Matching (FM)
- VP / DDPM-style perturbation paths

Its coefficients are decided by both choices family and the  parameterization.

- We need to ultimatly decide the functional forms of the coefficient funcions a, b, c and d and w.


consider the general interpolation function 
z_t = a(t) x + b(t) eps

let r_hat_theta_t be the predicted output of the model at time t

r_hat(t, z_; theta) = NN(z_t, t; theta)

prediction space parameterization refers to when if we want the network to 
estimate one of : x, eps, v_t. 
(where v_t = c(t) x + d(t) eps, and v_t = d/dt of z_t = a(t) x + b(t) eps)



say we have a NN with x predict output, ie 
NN(z_t, t; theta) = r_hat(z_t, t; theta) = x_hat 

given the equation z_t = a(t) x + b(t) eps, we can convert an 
x_hat into an eps_hat: 
eps_hat = (z_t - a(t) x_hat) / b(t)

also vica-versa, 

what about converting an x_hat into a v_hat? 
well: given that 
v_t = c(t) x + d(t) eps
and z_t = a(t) x + b(t) eps
and v_t = dz_t / dt = c(t) a(t) x + d(t) b(t) eps = a'(t) x + b'(t) eps
we can write: 
v_hat = c(t) x_hat + d(t) eps_hat
     = c(t) x_hat + d(t) (z_t - a(t) x_hat) / b(t)
     = (c(t) x_hat b(t) + d(t) z_t - d(t) a(t) x_hat) / b(t)
     = (x_hat (c(t) b(t) - d(t) a(t)) + d(t) z_t) / b(t)

"""

#  network output conversion functions


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
        z_t = a(t) x + b(t) eps           (interpolation)
        v_t = a'(t) x + b'(t) eps         (ODE velocity)
    """

    def __init__(self, config: GDFMConfig):
        super().__init__()
        self.config = config

        # these two sets up all the class attributes
        # ie: a, b, w, _sample_t, family_name,
        self._setup_family(config)
        self._setup_network(config)

    # --------------------------------------------------------
    # Family setup
    # --------------------------------------------------------
    # --------------------------------------------------------
    # Family setup
    #
    # Adding a new family (e.g. sub-VP, cosine schedule):
    #   1. Implement _a_new, _b_new, _da_dt_new, _db_dt_new
    #   2. Add a branch here that wires them up
    #   3. That's it — loss, conversion, sampling all inherit
    # --------------------------------------------------------
    def _setup_family(self, cfg):
        self.family_name = cfg.family

        if cfg.family == "fm":
            self.a = self._a_fm
            self.b = self._b_fm
            self.da_dt = self._da_dt_fm
            self.db_dt = self._db_dt_fm
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
            self.da_dt = self._da_dt_vp
            self.db_dt = self._db_dt_vp
            self.w = self._w_const
            self._sample_t = self._sample_t_uniform

        else:
            raise ValueError(f"Unknown family: {cfg.family}")

    # --------------------------------------------------------
    # Network setup
    # --------------------------------------------------------

    def _setup_network(self, cfg):
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
    # Prediction and loss computation
    # --------------------------------------------------------

    def predict(self, z, t):
        return self.net(z, t)

    # --------------------------------------------------------

    # Sampling
    # --------------------------------------------------------
    # --------------------------------------------------------
    # Sampling (ODE integration from noise to data)
    #
    # All sampling goes through _velocity(), which:
    #   1. Runs the network in prediction_space
    #   2. Converts to velocity (dz/dt) via convert()
    #
    # When prediction_space == "v", step 2 is identity since
    # our v IS the ODE velocity by construction.
    #
    # NFE (number of function evaluations) comparison:
    #   Euler:  n_steps NFE
    #   Heun:   2 * n_steps NFE
    # For fair comparison at equal NFE: euler(100) vs heun(50).
    #
    # The time grid runs from t_min to t_max, NOT 0 to 1.
    # This avoids the endpoint singularities in the converters.
    # --------------------------------------------------------
    @torch.no_grad()
    def sample(self, z, n_steps=100, method="euler", device=None):
        """
        Generate samples by integrating the ODE from noise (t=0) to data (t=1).

        shape:   shape of each sample, e.g. (D,) for flat or (C, H, W) for images
        n_steps: number of integration steps
        method:  "euler" or "heun"
        device:  defaults to model device

        Returns: [n_samples, ...] tensor of generated samples
                where n_samples = shape[0] if shape includes batch dim,
                or pass shape = (B, D) / (B, C, H, W) explicitly.
        """
        if device is None:
            device = next(self.parameters()).device

        # time grid from t_min to t_max
        t_min, t_max = self.config.t_min, self.config.t_max
        ts = torch.linspace(t_min, t_max, n_steps + 1, device=device)

        if method == "euler":
            z = self._sample_euler(z, ts)
        elif method == "heun":
            z = self._sample_heun(z, ts)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return z

    def _velocity(self, z, t_scalar, B):
        """
        Evaluate the learned velocity field at (z, t).

        Predicts in prediction_space, converts to velocity (dz/dt).
        When prediction_space == "v", convert is identity since
        our v-target IS the ODE velocity by construction.
        """
        t = torch.full((B,), t_scalar, device=z.device)
        r_hat = self.predict(z, t)
        return self.convert(
            r_hat, z, t, source=self.config.prediction_space, target="v"
        )

    def _sample_euler(self, z, ts):
        """
        Euler integrator:
            z_{k+1} = z_k + dt * v(z_k, t_k)
        """
        B = z.shape[0]
        for k in range(len(ts) - 1):
            dt = ts[k + 1] - ts[k]
            v = self._velocity(z, ts[k].item(), B)
            z = z + dt * v
        return z

    def _sample_heun(self, z, ts):
        """
        Heun integrator (2nd order, 2 NFE per step):
            v1 = v(z_k, t_k)
            z_tilde = z_k + dt * v1
            v2 = v(z_tilde, t_{k+1})
            z_{k+1} = z_k + dt/2 * (v1 + v2)
        """
        B = z.shape[0]
        for k in range(len(ts) - 1):
            dt = ts[k + 1] - ts[k]
            v1 = self._velocity(z, ts[k].item(), B)
            z_tilde = z + dt * v1
            v2 = self._velocity(z_tilde, ts[k + 1].item(), B)
            z = z + 0.5 * dt * (v1 + v2)
        return z

    # --------------------------------------------------------
    # Loss computation
    #
    # The training step is:
    #   1. Sample noise eps and time t
    #   2. Build noisy state z_t = a(t)x + b(t)eps
    #   3. Network predicts r_hat in prediction_space
    #   4. Convert r_hat to loss_space (may be identity)
    #   5. Compare against ground truth in loss_space
    #
    # When prediction_space == loss_space, step 4 is a no-op.
    # When they differ, convert() does the algebra using z_t.
    #
    # The weighting w(t) is currently constant (=1). To add
    # SNR weighting or min-SNR, implement a new _w_* method
    # and wire it in _setup_family.
    # --------------------------------------------------------
    def compute_loss(self, x, eps=None, t=None):
        """
        Full training-step loss:
            1. Sample eps ~ N(0,I) and t ~ p(t) if not provided
            2. Build z_t = a(t) x + b(t) eps
            3. Predict in prediction_space
            4. Convert to loss_space
            5. Compare against ground-truth target in loss_space
        """
        B, device = x.shape[0], x.device

        if eps is None:
            eps = torch.randn_like(x)
        if t is None:
            t = self.sample_t(B, device)

        z_t = self.z_t_interpolate(x, eps, t)
        r_hat = self.predict(z_t, t)

        r_hat_loss = self.convert(
            r_hat,
            z_t,
            t,
            source=self.config.prediction_space,
            target=self.config.loss_space,
        )

        r_true = self.get_true_target(x, eps, t, self.config.loss_space)

        t_b = self._bcast(t, x)
        w = self.w(t_b)
        per_sample = (w * (r_hat_loss - r_true) ** 2).flatten(1).mean(1)
        return per_sample.mean()

    # --------------------------------------------------------
    # Family-specific time sampling p(t)
    # --------------------------------------------------------

    def _sample_t_uniform(self, batch_size, device):
        """
        Uniform time sampling:
            t ~ Uniform(t_min, t_max)
        """
        cfg = self.config
        return (
            torch.rand(batch_size, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min
        )

    def _sample_t_logit_normal(self, batch_size, device):
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
    # Time and path
    # --------------------------------------------------------

    def sample_t(self, batch_size, device):
        return self._sample_t(batch_size, device)

    def z_t_interpolate(self, x, eps, t):
        """
        z_t = a(t) x + b(t) eps
        """
        t_b = self._bcast(t, x)
        return self.a(t_b) * x + self.b(t_b) * eps

    # --------------------------------------------------------
    # FM coefficients
    # --------------------------------------------------------

    def _a_fm(self, t):
        """
        FM signal coefficient:
            a(t) = t
        """
        return t

    def _b_fm(self, t):
        """
        FM noise coefficient:
            b(t) = 1 - t
        """
        return 1.0 - t

    # --------------------------------------------------------
    # FM derivatives
    # --------------------------------------------------------

    def _da_dt_fm(self, t):
        """a(t) = t  →  a'(t) = 1"""
        return torch.ones_like(t)

    def _db_dt_fm(self, t):
        """b(t) = 1-t  →  b'(t) = -1"""
        return -torch.ones_like(t)

    # --------------------------------------------------------
    # VP coefficients
    # --------------------------------------------------------

    def _a_vp(self, t):
        """
        VP data coefficient under the convention:
            t = 0 -> noise end
            t = 1 -> data end

        We map to standard VP time via
            s = 1 - t

        With linear beta schedule
            beta(s) = beta_min + (beta_max - beta_min) s

        define
            alpha_bar(s) = exp(- integral_0^s beta(u) du)

        Then
            a(t) = sqrt(alpha_bar(1 - t)).
        """
        s = 1.0 - t
        bmin, bmax = self.config.beta_min, self.config.beta_max
        int_beta = bmin * s + 0.5 * (bmax - bmin) * s**2
        return torch.exp(-0.5 * int_beta)

    def _b_vp(self, t):
        """
        VP noise coefficient:
            b(t) = sqrt(1 - a(t)^2)

        so that
            z_t = a(t) x + b(t) eps
        interpolates between data and noise in the standard VP way.
        """
        a = self._a_vp(t)
        return torch.sqrt(torch.clamp(1.0 - a**2, min=0.0))

    # --------------------------------------------------------
    # VP derivatives
    # --------------------------------------------------------

    def _beta_vp(self, t):
        """
        Instantaneous noise rate at our time t.

        beta(s) = beta_min + (beta_max - beta_min)*s
        evaluated at s = 1-t:
            beta(1-t) = beta_min + (beta_max - beta_min)*(1-t)
        """
        bmin, bmax = self.config.beta_min, self.config.beta_max
        return bmin + (bmax - bmin) * (1.0 - t)

    def _da_dt_vp(self, t):
        """
        a'(t) = beta(1-t)/2 * a(t)

        Derived from a(t) = sqrt(alpha_bar(1-t)), chain rule
        through s = 1-t. See Appendix A.
        """
        return 0.5 * self._beta_vp(t) * self._a_vp(t)

    # def _db_dt_vp(self, t):
    #     """
    #     b'(t) = -beta(1-t) * a(t)^2 / (2 * b(t))

    #     Derived from b = sqrt(1 - a^2), chain rule.
    #     See Appendix A.
    #     """
    #     a = self._a_vp(t)
    #     b = self._b_vp(t)
    #     beta = self._beta_vp(t)
    #     return -beta * a**2 / (2.0 * b)
    def _db_dt_vp(self, t):
        """b'(t) = -beta(1-t) * a(t)^2 / (2 * b(t))"""
        a = self._a_vp(t)
        b = self._b_vp(t)
        beta = self._beta_vp(t)
        return self._safe_div(-beta * a**2, 2.0 * b)

    # --------------------------------------------------------
    # Weighting w(t)
    # --------------------------------------------------------

    def _w_const(self, t):
        """
        Constant weighting:
            w(t) = 1
        """
        return torch.ones_like(t)

    def get_true_target(self, x, eps, t, space):
        """
        Ground-truth regression target in the chosen space.

            "x"   → r = x
            "eps" → r = eps
            "v"   → r = a'(t) x + b'(t) eps    (ODE velocity dz_t/dt)

        Only called during training, where we have ground-truth x and eps.
        """
        if space == "x":
            return x
        elif space == "eps":
            return eps
        elif space == "v":
            t_b = self._bcast(t, x)
            return self.da_dt(t_b) * x + self.db_dt(t_b) * eps
        else:
            raise ValueError(f"Unknown target space: {space}")

    # --------------------------------------------------------
    # Network output conversion helpers
    #
    # All derived from z_t = a(t)x + b(t)eps and v_t = a'(t)x + b'(t)eps.
    # Every conversion routes through recovering x_hat and eps_hat.
    # --------------------------------------------------------

    # def _x_hat_to_eps_hat(self, x_hat, z, t):
    #     """eps_hat = (z - a*x_hat) / b"""
    #     return (z - self.a(t) * x_hat) / self.b(t)
    # safe version
    def _x_hat_to_eps_hat(self, x_hat, z, t):
        """eps_hat = (z - a*x_hat) / b"""
        return self._safe_div(z - self.a(t) * x_hat, self.b(t))

    # def _eps_hat_to_x_hat(self, eps_hat, z, t):
    #     """x_hat = (z - b*eps_hat) / a"""
    #     return (z - self.b(t) * eps_hat) / self.a(t)

    def _eps_hat_to_x_hat(self, eps_hat, z, t):
        """x_hat = (z - b*eps_hat) / a"""
        return self._safe_div(z - self.b(t) * eps_hat, self.a(t))

    def _x_hat_to_v_hat(self, x_hat, z, t):
        """v_hat = a'*x_hat + b'*eps_hat, with eps_hat from x_hat"""
        eps_hat = self._x_hat_to_eps_hat(x_hat, z, t)
        return self.da_dt(t) * x_hat + self.db_dt(t) * eps_hat

    def _eps_hat_to_v_hat(self, eps_hat, z, t):
        """v_hat = a'*x_hat + b'*eps_hat, with x_hat from eps_hat"""
        x_hat = self._eps_hat_to_x_hat(eps_hat, z, t)
        return self.da_dt(t) * x_hat + self.db_dt(t) * eps_hat

    # def _v_hat_to_x_hat(self, v_hat, z, t):
    #     """
    #     From v = a'x + b'eps and z = ax + beps, eliminate eps:
    #         x_hat = (b*v_hat - b'*z) / (a'*b - a*b')
    #     """
    #     a, b = self.a(t), self.b(t)
    #     da, db = self.da_dt(t), self.db_dt(t)
    #     return (b * v_hat - db * z) / (da * b - a * db)
    def _v_hat_to_x_hat(self, v_hat, z, t):
        """x_hat = (b*v_hat - b'*z) / (a'*b - a*b')"""
        a, b = self.a(t), self.b(t)
        da, db = self.da_dt(t), self.db_dt(t)
        return self._safe_div(b * v_hat - db * z, da * b - a * db)

    def _v_hat_to_eps_hat(self, v_hat, z, t):
        """Route through x_hat"""
        x_hat = self._v_hat_to_x_hat(v_hat, z, t)
        return self._x_hat_to_eps_hat(x_hat, z, t)

    def convert(self, r_hat, z_t, t, source, target):
        """
        Convert network output r_hat from source space to target space.

        r_hat:  network prediction (x_hat, eps_hat, or v_hat depending on source)
        z_t:    noisy interpolant
        t:      diffusion time [B]
        source: what the network predicted ("x", "eps", "v")
        target: what we need ("x", "eps", "v")
        """
        if source == target:
            return r_hat

        t_b = self._bcast(t, z_t)

        key = (source, target)
        _converters = {
            ("x", "eps"): self._x_hat_to_eps_hat,
            ("x", "v"): self._x_hat_to_v_hat,
            ("eps", "x"): self._eps_hat_to_x_hat,
            ("eps", "v"): self._eps_hat_to_v_hat,
            ("v", "x"): self._v_hat_to_x_hat,
            ("v", "eps"): self._v_hat_to_eps_hat,
        }

        if key not in _converters:
            raise ValueError(f"Unknown conversion: {source} → {target}")

        return _converters[key](r_hat, z_t, t_b)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    # def _safe_div(self, num, den):
    #     """Clamp denominator away from zero to avoid numerical blowup."""
    #     sign = den.sign()
    #     sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    #     safe_den = sign * den.abs().clamp(min=1e-7)
    #     return num / safe_den

    def _safe_div(self, num, den):
        """Clamp denominator away from zero.

        Floor value follows Li & He (2025): 0.05 prevents
        gradient amplification beyond ~20x near time endpoints.
        For 2D synthetic experiments a smaller value (1e-3) is
        fine; for images 0.05 is empirically necessary.
        """
        sign = den.sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        safe_den = sign * den.abs().clamp(min=self.config.safe_div_eps)
        return num / safe_den

    # --------------------------------------------------------
    # Shape helpers
    #
    # _bcast handles the shape mismatch between t [B] and
    # data tensors [B, D] or [B, C, H, W]. It reshapes t to
    # [B, 1, ...] so that scalar-per-sample multiplication
    # broadcasts correctly regardless of data shape.
    # --------------------------------------------------------

    def _bcast(self, t, ref):
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

    def __repr__(self):
        return (
            "GDFM(\n"
            f"  family           = {self.family_name}\n"
            f"  prediction_space = {self.config.prediction_space}\n"
            f"  loss_space       = {self.config.loss_space}\n"
            f"  params           = {self.n_params():,}\n"
            f"  network          = {self.config.network}\n"
            ")"
        )


## Better print model summary


def print_model_summary(model):
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    print(model)  # your custom __repr__

    print("\n--- Network ---")
    print(model.net)

    print("\n--- Parameter counts ---")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total     : {total:,}")
    print(f"Trainable : {trainable:,}")

    print("\n--- Layer-wise params ---")
    for name, p in model.named_parameters():
        print(f"{name:40s} {tuple(p.shape)}  ({p.numel():,})")


def print_forward_shapes(model, x, t):
    print("=" * 60)
    print("FORWARD SHAPES")
    print("=" * 60)

    def hook(name):
        def fn(module, inp, out):
            in_shape = tuple(inp[0].shape) if isinstance(inp, tuple) else "?"
            out_shape = tuple(out.shape)
            print(f"{name:30s} {in_shape} -> {out_shape}")

        return fn

    handles = []
    for name, module in model.net.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook(name)))

    z = x
    _ = model.net(z, t)

    for h in handles:
        h.remove()
