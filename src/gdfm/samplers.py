"""
External sampler sidecar for GDFM.

Keep the working samplers in `gdfm.core` for now. This file mirrors them from
the outside so we can test sampler interfaces without changing the model
class.
"""

import torch


def _get_value(container, name, default=None):
    """Read one field from a mapping-like config or an attribute object."""
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


def _named_section(cfg, *names):
    """Return the first existing subsection among `names`."""
    for name in names:
        value = _get_value(cfg, name, None)
        if value is not None:
            return value
    return None


def _cfg_value(model, aliases, default=None):
    """Resolve one field from the model or its config."""
    for alias in aliases:
        value = getattr(model, alias, None)
        if value is not None:
            return value

    cfg = getattr(model, "config", None)
    sections = [
        _named_section(cfg, "framework"),
        _named_section(cfg, "schedule"),
        _named_section(cfg, "sampling"),
        cfg,
    ]

    for source in sections:
        for alias in aliases:
            value = _get_value(source, alias, None)
            if value is not None:
                return value

    return default


def _infer_device(model, z, device):
    """Choose the device used by the sampler."""
    if device is not None:
        return torch.device(device)
    if z is not None:
        return z.device
    return next(model.parameters()).device


def _batch_time(t, ref):
    """Normalize scalar or batched times to shape `[B]`."""
    if torch.is_tensor(t):
        if t.dim() == 0:
            return t.to(device=ref.device, dtype=ref.dtype).repeat(ref.shape[0])
        if t.dim() == 1:
            if t.shape[0] != ref.shape[0]:
                raise ValueError(
                    f"Batch mismatch: ref has B={ref.shape[0]}, t has B={t.shape[0]}"
                )
            return t.to(device=ref.device, dtype=ref.dtype)
        raise ValueError(f"Expected scalar t or [B] time tensor, got {tuple(t.shape)}")

    return torch.full((ref.shape[0],), float(t), device=ref.device, dtype=ref.dtype)


def _bcast(t, ref):
    """Broadcast a batch of scalar times to the rank of `ref`."""
    if t.dim() != 1:
        raise ValueError(f"Expected t to have shape [B], got {tuple(t.shape)}")
    return t.view([t.shape[0]] + [1] * (ref.dim() - 1))


def _safe_div(num, den, denom_eps):
    """Divide while clamping the denominator away from zero."""
    sign = den.sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    safe_den = sign * den.abs().clamp(min=denom_eps)
    return num / safe_den


def time_grid(model, n_steps=100, device=None, t_min=None, t_max=None):
    """Build the integration grid on the model's time interval."""
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    device = _infer_device(model, None, device)
    if t_min is None:
        t_min = _cfg_value(model, ("t_min",), default=1.0e-3)
    if t_max is None:
        t_max = _cfg_value(model, ("t_max",), default=0.999)

    return torch.linspace(float(t_min), float(t_max), n_steps + 1, device=device)


def predict_in_prediction_space(model, z, t):
    """Evaluate the model in its native prediction space."""
    t = _batch_time(t, z)
    return model.predict(z, t)


def convert_prediction(model, r_hat, z_t, t, target):
    """Convert a model prediction into the requested target space."""
    t = _batch_time(t, z_t)
    return model.convert(
        r_hat,
        z_t,
        t,
        source=_cfg_value(model, ("prediction_space",), default="v"),
        target=target,
    )


def model_to_x(model, z_t, t):
    """Convert the model prediction at `(z_t, t)` into `x` space."""
    r_hat = predict_in_prediction_space(model, z_t, t)
    return convert_prediction(model, r_hat, z_t, t, target="x")


def model_to_eps(model, z_t, t):
    """Convert the model prediction at `(z_t, t)` into `eps` space."""
    r_hat = predict_in_prediction_space(model, z_t, t)
    return convert_prediction(model, r_hat, z_t, t, target="eps")


def model_to_velocity(model, z_t, t):
    """Convert the model prediction at `(z_t, t)` into velocity space."""
    r_hat = predict_in_prediction_space(model, z_t, t)
    return convert_prediction(model, r_hat, z_t, t, target="v")


def model_to_score(model, z_t, t):
    """
    Convert the model prediction at `(z_t, t)` into a VP score estimate.

    Under `z_t = a(t) x + b(t) eps`, the score is `-eps_hat / b(t)`.
    """
    t = _batch_time(t, z_t)
    t_b = _bcast(t, z_t)
    eps_hat = model_to_eps(model, z_t, t)
    denom_eps = _cfg_value(model, ("safe_div_eps", "denom_eps"), default=0.05)
    return _safe_div(-eps_hat, model.b(t_b), float(denom_eps))


def sample_euler(model, z, n_steps=100, device=None, t_min=None, t_max=None):
    """Integrate the learned ODE with forward Euler."""
    device = _infer_device(model, z, device)
    z = z.to(device)
    ts = time_grid(model, n_steps=n_steps, device=device, t_min=t_min, t_max=t_max)

    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]
        v_t = model_to_velocity(model, z, ts[k])
        z = z + dt * v_t

    return z


def sample_heun(model, z, n_steps=100, device=None, t_min=None, t_max=None):
    """Integrate the learned ODE with Heun's method."""
    device = _infer_device(model, z, device)
    z = z.to(device)
    ts = time_grid(model, n_steps=n_steps, device=device, t_min=t_min, t_max=t_max)

    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]
        v_1 = model_to_velocity(model, z, ts[k])
        z_tilde = z + dt * v_1
        v_2 = model_to_velocity(model, z_tilde, ts[k + 1])
        z = z + 0.5 * dt * (v_1 + v_2)

    return z


def vp_beta(model, z_t, t):
    """Evaluate the VP beta schedule on the model's time convention."""
    if _cfg_value(model, ("family_name", "family"), default=None) != "vp":
        raise ValueError("vp_beta is only defined for VP-family models.")

    t = _batch_time(t, z_t)
    t_b = _bcast(t, z_t)
    return model._beta_vp(t_b)


def vp_probability_flow_drift(model, z_t, t):
    """
    Compute the VP probability-flow drift.

    Under the repo's `t=0 -> noise` convention, this should match the GDFM
    velocity away from the clamped endpoint region.
    """
    beta_t = vp_beta(model, z_t, t)
    score_t = model_to_score(model, z_t, t)
    return 0.5 * beta_t * z_t + 0.5 * beta_t * score_t


def vp_reverse_sde_drift(model, z_t, t):
    """Compute the reverse-time VP SDE drift."""
    beta_t = vp_beta(model, z_t, t)
    score_t = model_to_score(model, z_t, t)
    return 0.5 * beta_t * z_t + beta_t * score_t


def vp_reverse_sde_diffusion(model, z_t, t):
    """Compute the isotropic VP diffusion coefficient."""
    beta_t = vp_beta(model, z_t, t)
    return torch.sqrt(torch.clamp(beta_t, min=0.0))


def sample_vp_reverse_sde(
    model,
    z,
    n_steps=100,
    device=None,
    t_min=None,
    t_max=None,
    noise_scale=1.0,
):
    """Integrate the VP reverse SDE with Euler-Maruyama."""
    if _cfg_value(model, ("family_name", "family"), default=None) != "vp":
        raise ValueError("sample_vp_reverse_sde is only defined for VP-family models.")

    device = _infer_device(model, z, device)
    z = z.to(device)
    ts = time_grid(model, n_steps=n_steps, device=device, t_min=t_min, t_max=t_max)

    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]
        drift_t = vp_reverse_sde_drift(model, z, ts[k])
        diffusion_t = vp_reverse_sde_diffusion(model, z, ts[k])

        if k == len(ts) - 2:
            noise = torch.zeros_like(z)
        else:
            noise = torch.randn_like(z)

        z = z + dt * drift_t + noise_scale * torch.sqrt(dt) * diffusion_t * noise

    return z


def sample(model, z, n_steps=100, method="euler", device=None, t_min=None, t_max=None):
    """Dispatch to the requested external sampler."""
    if method == "euler":
        return sample_euler(
            model, z, n_steps=n_steps, device=device, t_min=t_min, t_max=t_max
        )
    if method == "heun":
        return sample_heun(
            model, z, n_steps=n_steps, device=device, t_min=t_min, t_max=t_max
        )
    if method == "vp_sde":
        return sample_vp_reverse_sde(
            model, z, n_steps=n_steps, device=device, t_min=t_min, t_max=t_max
        )

    raise ValueError(f"Unknown sampling method: {method}")


__all__ = [
    "convert_prediction",
    "model_to_eps",
    "model_to_score",
    "model_to_velocity",
    "model_to_x",
    "predict_in_prediction_space",
    "sample",
    "sample_euler",
    "sample_heun",
    "sample_vp_reverse_sde",
    "time_grid",
    "vp_beta",
    "vp_probability_flow_drift",
    "vp_reverse_sde_diffusion",
    "vp_reverse_sde_drift",
]


if __name__ == "__main__":
    from mp_config import get_base_config
    from gdfm.core import GDFM, print_forward_shapes, print_model_summary

    torch.manual_seed(0)

    print("gdfm.samplers example")

    # Build a tiny VP model so the example can exercise both ODE-style and
    # reverse-SDE-style sampling paths.
    config = get_base_config()
    config.framework.family = "vp"
    config.framework.prediction_space = "eps"
    config.framework.loss_space = "eps"
    config.network.arch = "mlp"
    config.network.hidden_dim = 32
    config.network.n_layers = 2
    config.network.time_emb_dim = 16
    config.data.data_dim = 8
    config.data.image_size = 0
    config.data.image_channels = 0

    # Use one latent batch and one fixed time slice for all sampler checks.
    model = GDFM(config)
    z = torch.randn(4, config.data.data_dim)
    t = torch.full((z.shape[0],), 0.5)

    # Print the model and traced backbone shapes before comparing samplers.
    print_model_summary(model)
    print_forward_shapes(model, z, t)

    # `model_to_velocity` checks the model-space to velocity-space conversion.
    # Euler and Heun check deterministic ODE integration. The VP reverse SDE
    # call checks the stochastic sampling path used only in the VP family.
    velocity = model_to_velocity(model, z, t)
    samples_euler = sample_euler(model, z.clone(), n_steps=4)
    samples_heun = sample_heun(model, z.clone(), n_steps=4)
    samples_dispatch = sample(model, z.clone(), n_steps=4, method="heun")
    samples_vp_sde = sample_vp_reverse_sde(model, z.clone(), n_steps=4)

    # Matching batch shapes are the first sanity check for all sidecar samplers.
    print(f"model_to_velocity(z, t) : {tuple(velocity.shape)}")
    print(f"sample_euler(z)         : {tuple(samples_euler.shape)}")
    print(f"sample_heun(z)          : {tuple(samples_heun.shape)}")
    print(f"sample(..., heun)       : {tuple(samples_dispatch.shape)}")
    print(f"sample_vp_reverse_sde   : {tuple(samples_vp_sde.shape)}")
