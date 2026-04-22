"""
Canonical GDFM model wrapper.

This file is the main mathematical contract for the repo. It defines:

1. the interpolation path between noise and data;
2. how network outputs are converted between `x`, `eps`, and velocity `v`;
3. the training loss for one clean data batch;
4. deterministic ODE sampling from an initial noise tensor.

Backbones live in `gdfm.networks`. This wrapper decides what a backbone
prediction means.
"""

from collections.abc import Mapping

import torch
import torch.nn as nn

from gdfm.model_setup import build_network


def _get_value(container, name, default=None):
    """
    Read one field from either a mapping-like config or an attribute object.
    """
    if container is None:
        return default

    if isinstance(container, Mapping):
        return container.get(name, default)

    return getattr(container, name, default)


def _named_section(cfg, *names):
    """
    Return the first existing subsection among `names`.
    """
    for name in names:
        value = _get_value(cfg, name, None)
        if value is not None:
            return value
    return None


def _cfg_value(cfg, names, aliases=(), default=None, required=False):
    """
    Resolve one config field from nested or legacy layouts.
    """
    sections = [
        _named_section(cfg, "framework"),
        _named_section(cfg, "schedule"),
        _named_section(cfg, "training"),
        _named_section(cfg, "sampling"),
        _named_section(cfg, "outputs"),
        cfg,
    ]

    for section_name in names:
        section = _named_section(cfg, section_name)
        if section is not None:
            for alias in aliases:
                value = _get_value(section, alias, None)
                if value is not None:
                    return value

    for source in sections:
        for alias in aliases:
            value = _get_value(source, alias, None)
            if value is not None:
                return value

    if required:
        joined = ", ".join(aliases)
        raise ValueError(f"Missing required config field: {joined}")

    return default


class GDFM(nn.Module):
    """
    Generalized linear-target denoising model.

    Convention:
        t = 0  -> noise/source end
        t = 1  -> data/target end

    Core path:
        z_t = a(t) x + b(t) eps
        v_t = a'(t) x + b'(t) eps

    The network can predict `x`, `eps`, or `v`. The `convert(...)` method is
    the single place that translates among these spaces.
    """

    def __init__(self, config):
        """
        Build the model from one study config.

        First read the path and target conventions. Then build the backbone.
        """
        super().__init__()
        self.config = config

        # =========================================================================
        # Config and conventions
        # =========================================================================

        # ---------------------------------------------------------------------
        # Read the framework-level choices that define what the network
        # predicts and what target space the loss is computed in.
        # ---------------------------------------------------------------------
        self.family_name = _cfg_value(
            config, ("framework",), ("family",), required=True
        )
        self.prediction_space = _cfg_value(
            config, ("framework",), ("prediction_space",), required=True
        )
        self.loss_space = _cfg_value(
            config, ("framework",), ("loss_space",), required=True
        )

        self.t_min = _cfg_value(
            config, ("schedule",), ("t_min",), default=1.0e-3
        )
        self.t_max = _cfg_value(
            config, ("schedule",), ("t_max",), default=0.999
        )
        self.safe_div_eps = _cfg_value(
            config,
            ("schedule",),
            ("denom_eps", "safe_div_eps"),
            default=0.05,
        )
        self.beta_min = _cfg_value(
            config, ("schedule",), ("beta_min",), default=0.1
        )
        self.beta_max = _cfg_value(
            config, ("schedule",), ("beta_max",), default=20.0
        )
        self.fm_time_sampling = _cfg_value(
            config,
            ("schedule", "framework"),
            ("time_sampling", "fm_time_sampling"),
            default="uniform",
        )
        self.fm_logit_mu = _cfg_value(
            config,
            ("schedule", "framework"),
            ("logit_mu", "fm_logit_mu"),
            default=0.0,
        )
        self.fm_logit_sigma = _cfg_value(
            config,
            ("schedule", "framework"),
            ("logit_sigma", "fm_logit_sigma"),
            default=1.0,
        )

        # =========================================================================
        # Path family and backbone
        # =========================================================================

        self._setup_family()
        self.net = build_network(config)

    def _setup_family(self):
        """
        Bind the family-specific coefficient functions.

        After this, the rest of the class can use `a(t)`, `b(t)`, `a'(t)`,
        `b'(t)`, and the active `t` sampler without branching on the family.
        """
        if self.family_name == "fm":
            self.a = self._a_fm
            self.b = self._b_fm
            self.da_dt = self._da_dt_fm
            self.db_dt = self._db_dt_fm
            self.w = self._w_const

            if self.fm_time_sampling == "uniform":
                self._sample_t = self._sample_t_uniform
            elif self.fm_time_sampling == "logit_normal":
                self._sample_t = self._sample_t_logit_normal
            else:
                raise ValueError(
                    f"Unknown FM time sampling scheme: {self.fm_time_sampling}"
                )
            return

        if self.family_name == "vp":
            self.a = self._a_vp
            self.b = self._b_vp
            self.da_dt = self._da_dt_vp
            self.db_dt = self._db_dt_vp
            self.w = self._w_const
            self._sample_t = self._sample_t_uniform
            return

        raise ValueError(f"Unknown family: {self.family_name}")

    # =========================================================================
    # Prediction and sampling
    # =========================================================================

    def predict(self, z, t):
        """
        Evaluate the backbone at `(z_t, t)`.

        The returned tensor is in `self.prediction_space`, not necessarily in
        velocity space.
        """
        return self.net(z, t)

    @torch.no_grad()
    def sample(self, z, n_steps=100, method="euler", device=None):
        """
        Integrate the learned ODE from noise to data.

        Inputs:
            z: initial noise tensor, already in the ambient data space
            n_steps: number of solver steps between `t_min` and `t_max`
            method: `"euler"` or `"heun"`
        """
        if device is None:
            device = next(self.parameters()).device

        z = z.to(device)
        ts = torch.linspace(self.t_min, self.t_max, n_steps + 1, device=device)

        if method == "euler":
            return self._sample_euler(z, ts)
        if method == "heun":
            return self._sample_heun(z, ts)

        raise ValueError(f"Unknown sampling method: {method}")

    def _velocity(self, z, t_scalar, batch_size):
        """
        Convert the network prediction at time `t` into a velocity field.

        The solver always integrates `v_t`, even if the network predicts `x`
        or `eps`.
        """
        t = torch.full((batch_size,), t_scalar, device=z.device)
        prediction = self.predict(z, t)
        return self.convert(
            prediction,
            z,
            t,
            source=self.prediction_space,
            target="v",
        )

    def _sample_euler(self, z, ts):
        """
        Integrate the learned ODE with first-order forward Euler steps.

        Euler uses one network evaluation per step. It is simple and useful for
        sanity checks.
        """
        batch_size = z.shape[0]
        for index in range(len(ts) - 1):
            dt = ts[index + 1] - ts[index]
            velocity = self._velocity(z, ts[index].item(), batch_size)
            z = z + dt * velocity
        return z

    def _sample_heun(self, z, ts):
        """
        Integrate the learned ODE with Heun's predictor-corrector method.

        Heun uses two slope evaluations per step, then averages them. It costs
        more per step than Euler but usually gives a better deterministic
        trajectory.
        """
        batch_size = z.shape[0]
        for index in range(len(ts) - 1):
            dt = ts[index + 1] - ts[index]
            velocity_1 = self._velocity(z, ts[index].item(), batch_size)
            z_tilde = z + dt * velocity_1
            velocity_2 = self._velocity(z_tilde, ts[index + 1].item(), batch_size)
            z = z + 0.5 * dt * (velocity_1 + velocity_2)
        return z

    # =========================================================================
    # Loss and targets
    # =========================================================================

    def compute_loss(self, x, eps=None, t=None):
        """
        Compute the training loss for one batch of clean data samples `x`.

        Data flow:
            1. sample or accept a noise tensor `eps`
            2. sample or accept times `t`
            3. build the noisy interpolation `z_t`
            4. predict in the model's prediction space
            5. convert that prediction into the configured loss space
            6. compare against the true target in that same space
        """
        batch_size = x.shape[0]
        device = x.device

        if eps is None:
            eps = torch.randn_like(x)
        if t is None:
            t = self.sample_t(batch_size, device)

        z_t = self.z_t_interpolate(x, eps, t)
        prediction = self.predict(z_t, t)
        prediction_in_loss_space = self.convert(
            prediction,
            z_t,
            t,
            source=self.prediction_space,
            target=self.loss_space,
        )
        target = self.get_true_target(x, eps, t, self.loss_space)

        t_b = self._bcast(t, x)
        weight = self.w(t_b)
        per_sample = (weight * (prediction_in_loss_space - target) ** 2).flatten(1).mean(1)
        return per_sample.mean()

    def sample_t(self, batch_size, device):
        """
        Draw training-time scalar times from the active family rule.
        """
        return self._sample_t(batch_size, device)

    def z_t_interpolate(self, x, eps, t):
        """
        Build the path point `z_t = a(t) x + b(t) eps`.
        """
        t_b = self._bcast(t, x)
        return self.a(t_b) * x + self.b(t_b) * eps

    def _sample_t_uniform(self, batch_size, device):
        """
        Sample times uniformly from the active interval `[t_min, t_max]`.
        """
        return (
            torch.rand(batch_size, device=device) * (self.t_max - self.t_min)
            + self.t_min
        )

    def _sample_t_logit_normal(self, batch_size, device):
        """
        Sample times from a Gaussian in logit space.
        """
        noise = self.fm_logit_mu + self.fm_logit_sigma * torch.randn(
            batch_size, device=device
        )
        t = torch.sigmoid(noise)
        return t.clamp(self.t_min, self.t_max)

    # =========================================================================
    # FM family coefficients
    # =========================================================================

    def _a_fm(self, t):
        """Flow-matching path coefficient multiplying the clean data term."""
        return t

    def _b_fm(self, t):
        """Flow-matching path coefficient multiplying the noise term."""
        return 1.0 - t

    def _da_dt_fm(self, t):
        """Time derivative of `a(t)` for the FM family."""
        return torch.ones_like(t)

    def _db_dt_fm(self, t):
        """Time derivative of `b(t)` for the FM family."""
        return -torch.ones_like(t)

    # =========================================================================
    # VP family coefficients
    # =========================================================================

    def _a_vp(self, t):
        """
        VP signal coefficient.

        The repo's convention is always `t=0 -> noise`, `t=1 -> data`, so we
        translate the usual VP schedule into that direction here.
        """
        s = 1.0 - t
        int_beta = self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * s**2
        return torch.exp(-0.5 * int_beta)

    def _b_vp(self, t):
        """
        VP noise coefficient paired with `a(t)` through `a(t)^2 + b(t)^2 = 1`.
        """
        a = self._a_vp(t)
        return torch.sqrt(torch.clamp(1.0 - a**2, min=0.0))

    def _beta_vp(self, t):
        """
        Linear VP beta schedule evaluated under the `t=0 -> noise` convention.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * (1.0 - t)

    def _da_dt_vp(self, t):
        """
        Derivative of the VP signal coefficient under the repo's time convention.
        """
        return 0.5 * self._beta_vp(t) * self._a_vp(t)

    def _db_dt_vp(self, t):
        """
        Derivative of the VP noise coefficient.

        Use safe division near the endpoint where `b(t)` becomes small.
        """
        a = self._a_vp(t)
        b = self._b_vp(t)
        beta = self._beta_vp(t)
        return self._safe_div(-beta * a**2, 2.0 * b)

    def _w_const(self, t):
        """Constant unit loss weight used by the current first-party setups."""
        return torch.ones_like(t)

    # =========================================================================
    # Target-space conversions
    # =========================================================================

    def get_true_target(self, x, eps, t, space):
        """
        Return the exact training target in the requested representation space.

        If the loss is written in `x`, `eps`, or `v`, the target must be in
        that same space.
        """
        if space == "x":
            return x
        if space == "eps":
            return eps
        if space == "v":
            t_b = self._bcast(t, x)
            return self.da_dt(t_b) * x + self.db_dt(t_b) * eps
        raise ValueError(f"Unknown target space: {space}")

    def _x_hat_to_eps_hat(self, x_hat, z, t):
        """
        Convert an `x` prediction into an `eps` prediction at the same `z_t`.
        """
        return self._safe_div(z - self.a(t) * x_hat, self.b(t))

    def _eps_hat_to_x_hat(self, eps_hat, z, t):
        """
        Convert an `eps` prediction into an `x` prediction at the same `z_t`.
        """
        return self._safe_div(z - self.b(t) * eps_hat, self.a(t))

    def _x_hat_to_v_hat(self, x_hat, z, t):
        """
        Convert an `x` prediction into a velocity prediction.
        """
        eps_hat = self._x_hat_to_eps_hat(x_hat, z, t)
        return self.da_dt(t) * x_hat + self.db_dt(t) * eps_hat

    def _eps_hat_to_v_hat(self, eps_hat, z, t):
        """
        Convert an `eps` prediction into a velocity prediction.
        """
        x_hat = self._eps_hat_to_x_hat(eps_hat, z, t)
        return self.da_dt(t) * x_hat + self.db_dt(t) * eps_hat

    def _v_hat_to_x_hat(self, v_hat, z, t):
        """
        Recover an `x` prediction from a velocity prediction.
        """
        a = self.a(t)
        b = self.b(t)
        da = self.da_dt(t)
        db = self.db_dt(t)
        return self._safe_div(b * v_hat - db * z, da * b - a * db)

    def _v_hat_to_eps_hat(self, v_hat, z, t):
        """
        Recover an `eps` prediction from a velocity prediction.
        """
        x_hat = self._v_hat_to_x_hat(v_hat, z, t)
        return self._x_hat_to_eps_hat(x_hat, z, t)

    def convert(self, prediction, z_t, t, source, target):
        """
        Convert one prediction tensor between the supported target spaces.

        Supported spaces:
            - `x`
            - `eps`
            - `v`
        """
        if source == target:
            return prediction

        t_b = self._bcast(t, z_t)
        converters = {
            ("x", "eps"): self._x_hat_to_eps_hat,
            ("x", "v"): self._x_hat_to_v_hat,
            ("eps", "x"): self._eps_hat_to_x_hat,
            ("eps", "v"): self._eps_hat_to_v_hat,
            ("v", "x"): self._v_hat_to_x_hat,
            ("v", "eps"): self._v_hat_to_eps_hat,
        }

        key = (source, target)
        if key not in converters:
            raise ValueError(f"Unknown conversion: {source} -> {target}")

        return converters[key](prediction, z_t, t_b)

    def _safe_div(self, numerator, denominator):
        """
        Divide while protecting against vanishing denominators near endpoints.

        Preserve the sign and clamp the magnitude away from zero.
        """
        sign = denominator.sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        safe_denominator = sign * denominator.abs().clamp(min=self.safe_div_eps)
        return numerator / safe_denominator

    def _bcast(self, t, ref):
        """
        Broadcast a batch of scalar times to the rank of `ref`.
        """
        if t.dim() != 1:
            raise ValueError(f"Expected t shape [B], got {tuple(t.shape)}")
        if ref.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch mismatch: ref has {ref.shape[0]} samples, t has {t.shape[0]}"
            )
        return t.view([t.shape[0]] + [1] * (ref.dim() - 1))

    def n_params(self):
        """Return the total parameter count of the full model plus backbone."""
        return sum(parameter.numel() for parameter in self.parameters())

    def __repr__(self):
        """Return a short summary for logs and notebooks."""
        network_cfg = _named_section(self.config, "network")
        return (
            "GDFM(\n"
            f"  family           = {self.family_name}\n"
            f"  prediction_space = {self.prediction_space}\n"
            f"  loss_space       = {self.loss_space}\n"
            f"  params           = {self.n_params():,}\n"
            f"  network          = {network_cfg}\n"
            ")"
        )


def print_model_summary(model):
    """
    Print a compact structural summary for a GDFM model or raw backbone.

    If `model` has a `.net` attribute, we treat that as the backbone and print
    both the outer model and the inner network. Otherwise we treat `model`
    itself as the network being inspected.
    """
    backbone = getattr(model, "net", model)

    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(model)

    if backbone is not model:
        print("\n--- Network ---")
        print(backbone)

    print("\n--- Parameter counts ---")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Total     : {total:,}")
    print(f"Trainable : {trainable:,}")

    print("\n--- Layer-wise params ---")
    for name, parameter in model.named_parameters():
        print(f"{name:40s} {tuple(parameter.shape)}  ({parameter.numel():,})")


def print_forward_shapes(model, x, t):
    """
    Print layer-by-layer tensor shapes for the active backbone.

    For a full `GDFM` instance we trace `model.net`. For a raw backbone we
    trace the module directly.
    """
    backbone = getattr(model, "net", model)

    print("=" * 60)
    print("FORWARD SHAPES")
    print("=" * 60)

    def hook(name):
        def wrapped(module, inputs, outputs):
            input_shape = tuple(inputs[0].shape) if isinstance(inputs, tuple) else "?"
            output_shape = tuple(outputs.shape)
            print(f"{name:30s} {input_shape} -> {output_shape}")

        return wrapped

    handles = []
    for name, module in backbone.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(hook(name)))

    with torch.no_grad():
        _ = backbone(x, t)

    for handle in handles:
        handle.remove()


__all__ = ["GDFM", "print_forward_shapes", "print_model_summary"]


if __name__ == "__main__":
    from mp_config import get_base_config

    torch.manual_seed(0)

    print("GDFM example")

    # Build a tiny MLP-backed GDFM model with the simplest FM configuration.
    config = get_base_config()
    config.framework.family = "fm"
    config.framework.prediction_space = "x"
    config.framework.loss_space = "x"
    config.network.arch = "mlp"
    config.network.hidden_dim = 32
    config.network.n_layers = 2
    config.network.time_emb_dim = 16
    config.data.data_dim = 8
    config.data.image_size = 0
    config.data.image_channels = 0

    # `x` is the clean batch used for prediction and loss checks. `t` is the
    # time vector for one direct prediction call. `z` is the latent/noise batch
    # used to check the sampling path.
    model = GDFM(config)
    x = torch.randn(4, config.data.data_dim)
    t = torch.linspace(0.1, 0.9, x.shape[0])
    z = torch.randn_like(x)

    # Check the three main public behaviors: direct prediction, training loss,
    # and numerical sampling from an initial latent batch.
    prediction = model.predict(x, t)
    loss = model.compute_loss(x)
    samples = model.sample(z, n_steps=4, method="heun")

    # The summary shows the resolved backbone and parameter count. The traced
    # shapes show how the backbone acts on one batch.
    print_model_summary(model)
    print_forward_shapes(model, x, t)

    # These lines are the fast contract checks for the main public methods.
    print(f"predict(x, t) shape : {tuple(prediction.shape)}")
    print(f"compute_loss(x)     : {float(loss.item()):.6f}")
    print(f"sample(z) shape     : {tuple(samples.shape)}")
