"""
External sampler sidecar for the current GDFM prototype.

Nothing is moved out of gdfm_components.py yet. The point of this module is to
prove the sampler-facing API first, while the current in-model samplers remain
the reference implementation.

This sidecar assumes the model already knows the path geometry:

    z_t = a(t) x + b(t) eps
    v_t = a'(t) x + b'(t) eps

The sampler only asks the model for a prediction, then converts that prediction
into the object the numerical method needs:

    Euler / Heun          -> velocity
    VP probability-flow   -> velocity = 0.5 * beta * (z + score)
    VP reverse SDE        -> drift + diffusion

The important separation is:

- prediction_space says what the network output means
- the sampler decides what field it wants to integrate

So the sidecar works through convert(...), not through hard-coded assumptions
about whether the model predicts x, eps, or v.
"""

import torch


def _infer_device(model, z, device):
    if device is not None:
        return torch.device(device)
    if z is not None:
        return z.device
    return next(model.parameters()).device


def _batch_time(t, ref):
    """
    Accept either a scalar time or a batch of times and normalize to shape [B].
    """
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
    if t.dim() != 1:
        raise ValueError(f"Expected t to have shape [B], got {tuple(t.shape)}")
    return t.view([t.shape[0]] + [1] * (ref.dim() - 1))


def _safe_div(num, den, denom_eps):
    sign = den.sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    safe_den = sign * den.abs().clamp(min=denom_eps)
    return num / safe_den


def time_grid(model, n_steps=100, device=None, t_min=None, t_max=None):
    """
    Build the integration grid on the model's own time convention.

    We keep the same endpoint trimming as the in-model sampler:
        t in [t_min, t_max], not exactly [0, 1]
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    device = _infer_device(model, None, device)

    if t_min is None:
        t_min = model.config.t_min
    if t_max is None:
        t_max = model.config.t_max

    return torch.linspace(t_min, t_max, n_steps + 1, device=device)


def predict_in_prediction_space(model, z, t):
    t = _batch_time(t, z)
    return model.predict(z, t)


def convert_prediction(model, r_hat, z_t, t, target):
    t = _batch_time(t, z_t)
    return model.convert(
        r_hat,
        z_t,
        t,
        source=model.config.prediction_space,
        target=target,
    )


def model_to_x(model, z_t, t):
    r_hat = predict_in_prediction_space(model, z_t, t)
    return convert_prediction(model, r_hat, z_t, t, target="x")


def model_to_eps(model, z_t, t):
    r_hat = predict_in_prediction_space(model, z_t, t)
    return convert_prediction(model, r_hat, z_t, t, target="eps")


def model_to_velocity(model, z_t, t):
    r_hat = predict_in_prediction_space(model, z_t, t)
    return convert_prediction(model, r_hat, z_t, t, target="v")


def model_to_score(model, z_t, t):
    """
    Score-like quantity from eps-prediction under z_t = a(t) x + b(t) eps:

        score(z_t, t) = grad_z log p_t(z_t) = -eps_hat / b(t)

    This is the quantity VP stochastic samplers want. We keep the same
    denominator clamp used elsewhere in GDFM so the sidecar inherits the same
    numerical guardrails near the time endpoints. That means the textbook
    identity is exact on the interior and only approximate once the clamp is
    active.
    """
    t = _batch_time(t, z_t)
    t_b = _bcast(t, z_t)
    eps_hat = model_to_eps(model, z_t, t)
    return _safe_div(-eps_hat, model.b(t_b), model.config.safe_div_eps)


def sample_euler(model, z, n_steps=100, device=None, t_min=None, t_max=None):
    """
    Euler integration of the learned ODE field.

        z_{k+1} = z_k + dt * v(z_k, t_k)
    """
    device = _infer_device(model, z, device)
    z = z.to(device)
    ts = time_grid(model, n_steps=n_steps, device=device, t_min=t_min, t_max=t_max)

    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]
        v_t = model_to_velocity(model, z, ts[k])
        z = z + dt * v_t

    return z


def sample_heun(model, z, n_steps=100, device=None, t_min=None, t_max=None):
    """
    Heun integration of the learned ODE field.

    This is the same two-stage predictor-corrector used inside the current
    prototype, but now expressed entirely through the external sampler bridge.
    """
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
    """
    VP beta evaluated on the current time convention t = 0 -> noise, t = 1 -> data.
    """
    if getattr(model, "family_name", None) != "vp":
        raise ValueError("vp_beta is only defined for VP-family models.")

    t = _batch_time(t, z_t)
    t_b = _bcast(t, z_t)
    return model._beta_vp(t_b)


def vp_probability_flow_drift(model, z_t, t):
    """
    Probability-flow ODE drift for VP in the GDFM time convention.

    Under t = 1 - s, the usual VP probability-flow ODE becomes

        dz_t / dt = 0.5 * beta(1-t) * z_t + 0.5 * beta(1-t) * score(z_t, t)

    and this should agree with the true GDFM velocity field a'(t)x + b'(t)eps
    away from the clamped endpoint region.
    """
    beta_t = vp_beta(model, z_t, t)
    score_t = model_to_score(model, z_t, t)
    return 0.5 * beta_t * z_t + 0.5 * beta_t * score_t


def vp_reverse_sde_drift(model, z_t, t):
    """
    Reverse-time VP SDE drift in the GDFM time convention.

    Relative to the probability-flow ODE, the score term doubles:

        drift = 0.5 * beta(1-t) * z_t + beta(1-t) * score(z_t, t)
    """
    beta_t = vp_beta(model, z_t, t)
    score_t = model_to_score(model, z_t, t)
    return 0.5 * beta_t * z_t + beta_t * score_t


def vp_reverse_sde_diffusion(model, z_t, t):
    """
    Isotropic VP diffusion coefficient:

        g(t) = sqrt(beta(1-t))
    """
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
    """
    Euler-Maruyama sampler for the VP reverse SDE.

    This is intentionally a sidecar prototype. It gives us a running stochastic
    sampler outside the model so we can validate the interface before deciding
    how broad the sampler package should become.
    """
    if getattr(model, "family_name", None) != "vp":
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
    """
    Small external sampler dispatcher.

    Supported methods right now:
        - euler
        - heun
        - vp_sde
    """
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
