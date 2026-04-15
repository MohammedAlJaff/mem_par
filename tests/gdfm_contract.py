"""
Notebook-inspired contract checks for GDFM-style models.

These helpers are derived from the sanity checks in prototyping/dmf.ipynb.
They are written so future GDFM implementations can reuse the same checks as
long as they expose the current public contract:

- sample_t(batch_size, device)
- z_t_interpolate(x, eps, t)
- predict(z, t)
- compute_loss(x, eps=None, t=None)
- get_true_target(x, eps, t, space)
- convert(r_hat, z_t, t, source, target)
- sample(z, n_steps, method, device=None)
"""

try:
    import torch
except ModuleNotFoundError:
    torch = None


SPACES = ("x", "eps", "v")
DIRECT_CONVERSIONS = (
    ("x", "eps"),
    ("x", "v"),
    ("eps", "x"),
    ("eps", "v"),
    ("v", "x"),
    ("v", "eps"),
)
ROUND_TRIPS = (
    ("x", "eps", "x"),
    ("x", "v", "x"),
    ("eps", "x", "eps"),
    ("eps", "v", "eps"),
    ("v", "x", "v"),
    ("v", "eps", "v"),
)


def max_abs_err(a, b):
    if torch is None:
        raise RuntimeError("torch is required for GDFM contract checks.")
    return float((a - b).abs().max().item())


def make_flat_batch(model, batch_size=16):
    if torch is None:
        raise RuntimeError("torch is required for GDFM contract checks.")
    data_dim = getattr(model.config, "data_dim", None)
    if data_dim is None:
        raise AssertionError("Model config must expose data_dim for flat-batch tests.")

    x = torch.randn(batch_size, data_dim)
    eps = torch.randn_like(x)
    t = model.sample_t(batch_size, x.device)
    z_t = model.z_t_interpolate(x, eps, t)
    return x, eps, t, z_t


def make_interior_flat_batch(model, batch_size=2048, margin=0.01):
    """
    Build a batch in the regime where the denominator clamp is inactive.

    The prototype intentionally clamps small denominators near the time
    endpoints. That keeps training and conversion numerically safe, but it also
    means exact algebraic identities are only expected to hold away from those
    endpoints.
    """
    if torch is None:
        raise RuntimeError("torch is required for GDFM contract checks.")

    data_dim = getattr(model.config, "data_dim", None)
    if data_dim is None:
        raise AssertionError("Model config must expose data_dim for flat-batch tests.")

    x = torch.randn(batch_size, data_dim)
    eps = torch.randn_like(x)
    t = torch.linspace(
        model.config.t_min,
        model.config.t_max,
        batch_size,
        device=x.device,
    )

    t_b = model._bcast(t, x)
    a = model.a(t_b).flatten(1).amin(dim=1)
    b = model.b(t_b).flatten(1).amin(dim=1)
    threshold = model.config.safe_div_eps + margin
    mask = (a >= threshold) & (b >= threshold)

    if not mask.any():
        raise AssertionError(
            "No interior times remained after excluding the clamped endpoint region."
        )

    x = x[mask]
    eps = eps[mask]
    t = t[mask]
    z_t = model.z_t_interpolate(x, eps, t)
    return x, eps, t, z_t


def assert_prediction_shape(model, batch_size=16):
    _, _, t, z_t = make_flat_batch(model, batch_size=batch_size)
    y = model.predict(z_t, t)

    if y.shape != z_t.shape:
        raise AssertionError(
            f"Prediction shape mismatch: got {tuple(y.shape)}, expected {tuple(z_t.shape)}"
        )


def assert_loss_backward(model, batch_size=16):
    if torch is None:
        raise RuntimeError("torch is required for GDFM contract checks.")
    model.zero_grad(set_to_none=True)
    x, _, _, _ = make_flat_batch(model, batch_size=batch_size)
    loss = model.compute_loss(x)

    if not torch.isfinite(loss):
        raise AssertionError(f"Loss is not finite: {loss}")

    loss.backward()

    has_grad = any(
        (param.grad is not None) and torch.isfinite(param.grad).all() and (param.grad.abs().sum() > 0)
        for param in model.parameters()
    )
    if not has_grad:
        raise AssertionError("No gradients flowed through the model.")


def assert_conversion_cycles(model, batch_size=2048, atol=1e-4):
    x_full, eps_full, t_full, z_full = make_flat_batch(model, batch_size=batch_size)

    # Across the full time domain, including near clamped endpoints, the
    # converters should remain finite even when exact equality no longer holds.
    for source, target in DIRECT_CONVERSIONS:
        converted = model.convert(
            model.get_true_target(x_full, eps_full, t_full, source),
            z_full,
            t_full,
            source=source,
            target=target,
        )
        if not torch.isfinite(converted).all():
            raise AssertionError(
                f"Direct conversion {source}->{target} produced non-finite values."
            )

    # Exact algebraic identities are checked only on interior times where the
    # safe denominator clamp is inactive.
    x, eps, t, z_t = make_interior_flat_batch(model, batch_size=batch_size)
    truth = {
        space: model.get_true_target(x, eps, t, space)
        for space in SPACES
    }

    worst_err = 0.0

    for source, target in DIRECT_CONVERSIONS:
        converted = model.convert(truth[source], z_t, t, source=source, target=target)
        err = max_abs_err(converted, truth[target])
        worst_err = max(worst_err, err)
        if err >= atol:
            raise AssertionError(
                f"Direct conversion {source}->{target} exceeded tolerance: {err:.3e} >= {atol:.3e}"
            )

    for source, mid, target in ROUND_TRIPS:
        mid_value = model.convert(truth[source], z_t, t, source=source, target=mid)
        cycled = model.convert(mid_value, z_t, t, source=mid, target=target)
        err = max_abs_err(cycled, truth[target])
        worst_err = max(worst_err, err)
        if err >= atol:
            raise AssertionError(
                f"Round-trip conversion {source}->{mid}->{target} exceeded tolerance: "
                f"{err:.3e} >= {atol:.3e}"
            )

    return worst_err


def assert_sampling_contract(model, batch_size=64, n_steps=20):
    if torch is None:
        raise RuntimeError("torch is required for GDFM contract checks.")
    data_dim = getattr(model.config, "data_dim", None)
    if data_dim is None:
        raise AssertionError("Model config must expose data_dim for sampling tests.")

    z = torch.randn(batch_size, data_dim)
    outputs = {}

    for method in ("euler", "heun"):
        out = model.sample(z.clone(), n_steps=n_steps, method=method, device="cpu")
        outputs[method] = out

        if out.shape != z.shape:
            raise AssertionError(
                f"Sampling with {method} changed shape: got {tuple(out.shape)}, expected {tuple(z.shape)}"
            )
        if not torch.isfinite(out).all():
            raise AssertionError(f"Sampling with {method} produced non-finite values.")

    return outputs
