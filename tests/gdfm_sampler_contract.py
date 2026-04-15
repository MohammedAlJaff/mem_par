"""
Contract checks for external GDFM sampler sidecars.

These tests reuse the notebook spirit from the model contract checks:
we verify algebraic identities first, then make sure numerical samplers keep
shape and finiteness under the current public model contract.
"""

try:
    import torch
except ModuleNotFoundError:
    torch = None


def max_abs_err(a, b):
    if torch is None:
        raise RuntimeError("torch is required for GDFM sampler checks.")
    return float((a - b).abs().max().item())


def interior_time_mask(model, ref, t, margin=0.01):
    """
    Mask out the endpoint region where the prototype's denominator clamp is active.
    """
    t_b = model._bcast(t, ref)
    a = model.a(t_b).flatten(1).amin(dim=1)
    b = model.b(t_b).flatten(1).amin(dim=1)
    threshold = model.config.safe_div_eps + margin
    return (a >= threshold) & (b >= threshold)


def assert_external_sampler_matches_builtin(
    model, sampler_module, method, batch_size=64, n_steps=20, atol=1e-6
):
    if torch is None:
        raise RuntimeError("torch is required for GDFM sampler checks.")

    data_dim = getattr(model.config, "data_dim", None)
    if data_dim is None:
        raise AssertionError("Model config must expose data_dim for sampler tests.")

    z = torch.randn(batch_size, data_dim)
    expected = model.sample(z.clone(), n_steps=n_steps, method=method, device="cpu")

    if method == "euler":
        actual = sampler_module.sample_euler(
            model, z.clone(), n_steps=n_steps, device="cpu"
        )
    elif method == "heun":
        actual = sampler_module.sample_heun(
            model, z.clone(), n_steps=n_steps, device="cpu"
        )
    else:
        raise ValueError(f"Unsupported builtin comparison method: {method}")

    err = max_abs_err(actual, expected)
    if err >= atol:
        raise AssertionError(
            f"External sampler {method} diverged from builtin path: "
            f"{err:.3e} >= {atol:.3e}"
        )

    return err


def assert_vp_probability_flow_matches_velocity(
    model, sampler_module, batch_size=128, atol=5e-5
):
    if torch is None:
        raise RuntimeError("torch is required for GDFM sampler checks.")

    data_dim = getattr(model.config, "data_dim", None)
    if data_dim is None:
        raise AssertionError("Model config must expose data_dim for sampler tests.")

    z = torch.randn(batch_size, data_dim)
    t = model.sample_t(batch_size, z.device)

    drift_full = sampler_module.vp_probability_flow_drift(model, z, t)
    if not torch.isfinite(drift_full).all():
        raise AssertionError("VP probability-flow drift produced non-finite values.")

    mask = interior_time_mask(model, z, t)
    if not mask.any():
        raise AssertionError(
            "No interior times remained for the VP probability-flow identity check."
        )

    z = z[mask]
    t = t[mask]

    v_from_model = sampler_module.model_to_velocity(model, z, t)
    v_from_vp_bridge = sampler_module.vp_probability_flow_drift(model, z, t)

    err = max_abs_err(v_from_model, v_from_vp_bridge)
    if err >= atol:
        raise AssertionError(
            f"VP probability-flow drift no longer matches model velocity: "
            f"{err:.3e} >= {atol:.3e}"
        )

    return err


def assert_vp_reverse_sde_contract(model, sampler_module, batch_size=64, n_steps=20):
    if torch is None:
        raise RuntimeError("torch is required for GDFM sampler checks.")

    data_dim = getattr(model.config, "data_dim", None)
    if data_dim is None:
        raise AssertionError("Model config must expose data_dim for sampler tests.")

    z = torch.randn(batch_size, data_dim)
    out = sampler_module.sample_vp_reverse_sde(
        model, z.clone(), n_steps=n_steps, device="cpu"
    )

    if out.shape != z.shape:
        raise AssertionError(
            f"VP reverse SDE changed shape: got {tuple(out.shape)}, expected {tuple(z.shape)}"
        )
    if not torch.isfinite(out).all():
        raise AssertionError("VP reverse SDE produced non-finite values.")

    return out
