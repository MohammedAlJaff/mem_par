"""
Flow matching with the full 3×3 prediction/loss space grid.

This is the CORE module of the codebase. Everything else supports this.

The 3×3 grid:
    Given the flow matching interpolant z_t = t·x + (1-t)·ε, the network
    can OUTPUT any of {x, ε, v} and the LOSS can be measured in any of {x, ε, v}.
    That's 9 combinations. Li & He (2025) showed that the prediction space
    (what the network outputs) matters much more than the loss space (where
    error is measured). Our paper asks whether this also affects memorization.

Conversion table (from network output to any target):
    Given z_t = t·x + (1-t)·ε and v = x - ε:

    From x_θ: ε_θ = (z_t - t·x_θ) / (1-t)    v_θ = (x_θ - z_t) / (1-t)
    From ε_θ: x_θ = (z_t - (1-t)·ε_θ) / t     v_θ = (z_t - ε_θ) / t
    From v_θ: x_θ = (1-t)·v_θ + z_t            ε_θ = z_t - t·v_θ

    IMPORTANT: the denominators (1-t) and t can be zero at the endpoints.
    We clip them at DENOM_CLIP = 0.05 following Li & He.

Time sampling:
    We use the logit-normal distribution for t, following Esser et al. (2024)
    and Li & He. This upweights intermediate timesteps where the denoising
    signal is strongest, and downweights the endpoints where the signal is
    either trivial (t≈1, input is nearly clean) or overwhelmed by noise (t≈0).

    t = sigmoid(μ + σ·z),  z ~ N(0,1)
    Default: μ = -0.8, σ = 0.8 (Li & He's setting)

    Alternative: uniform t ~ U[0,1]. Simpler but wastes training signal
    at endpoints. Not recommended.

Sampling (generation):
    Euler or Heun ODE integration from t=0 (noise) to t=1 (data).
    Heun is second-order and much more accurate at the same step count.
    Default: 100 Heun steps (Li & He use 50 Heun steps).
"""

import torch
import torch.nn as nn
from typing import Literal


# =============================================================================
# Constants
# =============================================================================

# Li & He clip the denominator (1-t) at 0.05 to avoid division by zero.
# This affects the conversion between prediction spaces near t=1.
DENOM_CLIP = 0.05  # HYPERPARAMETER: Li & He use 0.05
# Alt: 0.01 (less clipping, more numerically risky)
# Alt: 0.1  (more clipping, more information loss near t=1)

# Time clipping: avoid exact 0 and 1 where the interpolant is degenerate.
T_CLIP_MIN = 0.001  # HYPERPARAMETER
T_CLIP_MAX = 0.999

# Logit-normal time sampling parameters
TIME_MU = -0.8  # HYPERPARAMETER: Li & He default
TIME_SIGMA = 0.8  # HYPERPARAMETER: Li & He default
# μ = 0, σ = 1 gives more uniform coverage
# μ = -1.2, σ = 0.8 upweights low-noise regime (helps x-pred)


# =============================================================================
# Time sampling
# =============================================================================


def sample_logit_normal(
    batch_size: int,
    mu: float = TIME_MU,
    sigma: float = TIME_SIGMA,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample t from a logit-normal distribution.

    The logit-normal is defined by:
        z ~ N(μ, σ²)
        t = sigmoid(z)

    This gives a smooth distribution on (0, 1) that peaks near sigmoid(μ).
    With μ = -0.8, the peak is around t ≈ 0.31, which upweights the
    "middle" of the diffusion process where learning signal is richest.

    Returns:
        (B,) tensor of timesteps in [T_CLIP_MIN, T_CLIP_MAX]
    """
    z = torch.randn(batch_size, device=device)
    t = torch.sigmoid(mu + sigma * z)
    t = t.clamp(T_CLIP_MIN, T_CLIP_MAX)
    return t


# =============================================================================
# Prediction space conversions
# =============================================================================
# These functions convert between x, ε, v given the interpolant z_t and time t.
# They are the algebraic heart of the 3×3 grid.


def convert_pred_to_x(
    pred: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    pred_space: str,
) -> torch.Tensor:
    """Convert any prediction to x-space."""
    t = _reshape_t(t, pred)

    if pred_space == "x":
        return pred
    elif pred_space == "eps":
        # x = (z_t - (1-t)·ε) / t
        return (z_t - (1 - t) * pred) / t.clamp(min=DENOM_CLIP)
    elif pred_space == "v":
        # x = (1-t)·v + z_t
        return (1 - t) * pred + z_t
    else:
        raise ValueError(f"Unknown pred_space: {pred_space}")


def convert_pred_to_eps(
    pred: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    pred_space: str,
) -> torch.Tensor:
    """Convert any prediction to ε-space."""
    t = _reshape_t(t, pred)

    if pred_space == "eps":
        return pred
    elif pred_space == "x":
        # ε = (z_t - t·x) / (1-t)
        return (z_t - t * pred) / (1 - t).clamp(min=DENOM_CLIP)
    elif pred_space == "v":
        # ε = z_t - t·v
        return z_t - t * pred
    else:
        raise ValueError(f"Unknown pred_space: {pred_space}")


def convert_pred_to_v(
    pred: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    pred_space: str,
) -> torch.Tensor:
    """Convert any prediction to v-space."""
    t = _reshape_t(t, pred)

    if pred_space == "v":
        return pred
    elif pred_space == "x":
        # v = (x - z_t) / (1-t)
        return (pred - z_t) / (1 - t).clamp(min=DENOM_CLIP)
    elif pred_space == "eps":
        # v = (z_t - ε) / t
        return (z_t - pred) / t.clamp(min=DENOM_CLIP)
    else:
        raise ValueError(f"Unknown pred_space: {pred_space}")


def _reshape_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Reshape t for broadcasting.
    Input t: (B,) or (B,1)
    Output: (B, 1) for vectors, (B, 1, 1, 1) for images
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)
    # Add spatial dims for images: (B, 1) → (B, 1, 1, 1)
    while t.dim() < ref.dim():
        t = t.unsqueeze(-1)
    return t


# =============================================================================
# Main FlowMatching class
# =============================================================================


class FlowMatching(nn.Module):
    """
    Flow matching wrapper that implements the 3×3 prediction/loss grid.

    This wraps any network (MLP, UNet, JiT) and adds:
        1. Loss computation in the specified loss space
        2. Conversion between prediction spaces
        3. ODE sampling (Euler and Heun)

    The network itself is agnostic to parameterization — it just maps
    (z_t, t) → R^D. The FlowMatching wrapper determines what that
    output MEANS (x, ε, or v) and how the loss is computed.

    This separation is crucial for the paper: we can swap parameterization
    without touching the network architecture.

    Args:
        network: any nn.Module with forward(x, t) → prediction
        pred_space: what the network output represents ('x', 'eps', 'v')
        loss_space: where the MSE loss is computed ('x', 'eps', 'v')

    The 9 combinations and their meaning:
        pred=x, loss=x:   network predicts x, loss in x-space (standard x-pred)
        pred=x, loss=v:   network predicts x, loss in v-space (Li & He Algorithm 1)
        pred=x, loss=eps: network predicts x, loss in ε-space
        pred=eps, loss=*: network predicts ε (standard DDPM-style)
        pred=v, loss=*:   network predicts v (standard in SD3, Flux)

    Li & He's key finding: the COLUMN (pred_space) matters. The ROW
    (loss_space) matters much less. Our finding (hypothesized): the
    column also determines where the memorization boundary sits.
    """

    def __init__(
        self,
        network: nn.Module,
        pred_space: Literal["x", "eps", "v"] = "x",
        loss_space: Literal["x", "eps", "v"] = "v",
    ):
        super().__init__()
        self.network = network
        self.pred_space = pred_space
        self.loss_space = loss_space

    def compute_loss(self, x_clean: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss on a batch of clean data.

        Steps:
            1. Sample noise ε ~ N(0, I)
            2. Sample time t ~ logit-normal
            3. Form interpolant: z_t = t·x + (1-t)·ε
            4. Compute target in loss_space
            5. Network predicts in pred_space
            6. Convert prediction to loss_space
            7. MSE between prediction and target in loss_space

        Args:
            x_clean: (B, D) for synthetic or (B, C, H, W) for images
                     Clean data samples from the training set.

        Returns:
            scalar loss (mean over batch)
        """
        B = x_clean.shape[0]
        device = x_clean.device

        # Step 1: Sample noise
        eps = torch.randn_like(x_clean)

        # Step 2: Sample time
        t = sample_logit_normal(B, device=device)

        # Step 3: Form interpolant z_t = t·x + (1-t)·ε
        t_broad = _reshape_t(t, x_clean)
        z_t = t_broad * x_clean + (1 - t_broad) * eps

        # Step 4: Compute ground-truth target in loss_space
        target = self._compute_target(x_clean, eps, z_t, t, self.loss_space)

        # Step 5: Network prediction
        pred = self.network(z_t, t)

        # Step 6: Convert prediction from pred_space to loss_space
        pred_in_loss_space = self._convert(
            pred, z_t, t, self.pred_space, self.loss_space
        )

        # Step 7: MSE loss
        loss = torch.mean((pred_in_loss_space - target) ** 2)
        return loss

    def _compute_target(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        space: str,
    ) -> torch.Tensor:
        """Compute the ground-truth target in the specified space."""
        if space == "x":
            return x
        elif space == "eps":
            return eps
        elif space == "v":
            return x - eps  # v = x - ε for flow matching linear interpolant
        else:
            raise ValueError(f"Unknown space: {space}")

    def _convert(
        self,
        pred: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        from_space: str,
        to_space: str,
    ) -> torch.Tensor:
        """Convert prediction from one space to another."""
        if from_space == to_space:
            return pred

        # Route to the appropriate conversion
        if to_space == "x":
            return convert_pred_to_x(pred, z_t, t, from_space)
        elif to_space == "eps":
            return convert_pred_to_eps(pred, z_t, t, from_space)
        elif to_space == "v":
            return convert_pred_to_v(pred, z_t, t, from_space)
        else:
            raise ValueError(f"Unknown to_space: {to_space}")

    # =========================================================================
    # Sampling (generation)
    # =========================================================================

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        n_steps: int = 100,
        method: str = "heun",
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate samples by integrating the learned ODE from t=0 to t=1.

        The ODE is: dz/dt = v(z_t, t), where v is the velocity field.
        We always convert the network prediction to v-space for integration,
        regardless of pred_space.

        Args:
            shape: output shape — (B, D) for synthetic, (B, C, H, W) for images
            n_steps: number of integration steps (100 is safe, 50 is faster)
            method: 'euler' (first-order) or 'heun' (second-order)
                Heun is ~2× more expensive per step but much more accurate.
                At 50 Heun steps you get better results than 200 Euler steps.
            device: where to generate

        Returns:
            Generated samples, same shape as input
        """
        if method == "euler":
            return self._sample_euler(shape, n_steps, device)
        elif method == "heun":
            return self._sample_heun(shape, n_steps, device)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _get_velocity(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Get velocity v(z_t, t) from the network.
        Converts from whatever pred_space the network uses to v-space.
        """
        pred = self.network(z_t, t)
        return convert_pred_to_v(pred, z_t, t, self.pred_space)

    def _sample_euler(self, shape: tuple, n_steps: int, device: str) -> torch.Tensor:
        """
        Euler integration: z_{t+dt} = z_t + dt · v(z_t, t)

        First-order. Simple, stable, but requires many steps for accuracy.
        The error per step is O(dt²), total error is O(dt).
        """
        z = torch.randn(shape, device=device)  # Start from pure noise at t=0
        dt = 1.0 / n_steps
        t_values = torch.linspace(0, 1, n_steps + 1, device=device)

        for i in range(n_steps):
            t_curr = t_values[i]
            t_batch = torch.full((shape[0],), t_curr.item(), device=device)
            t_batch = t_batch.clamp(T_CLIP_MIN, T_CLIP_MAX)

            v = self._get_velocity(z, t_batch)
            z = z + dt * v

        return z

    def _sample_heun(self, shape: tuple, n_steps: int, device: str) -> torch.Tensor:
        """
        Heun's method (improved Euler / explicit trapezoidal):
            z_predict = z_t + dt · v(z_t, t)
            z_{t+dt} = z_t + dt/2 · [v(z_t, t) + v(z_predict, t+dt)]

        Second-order. Two function evaluations per step (so 2× the cost
        of Euler per step), but O(dt²) total error. At the same wall-clock
        time, Heun is much better than Euler.

        Li & He use 50 Heun steps for their best results.
        """
        z = torch.randn(shape, device=device)
        dt = 1.0 / n_steps
        t_values = torch.linspace(0, 1, n_steps + 1, device=device)

        for i in range(n_steps):
            t_curr = t_values[i].item()
            t_next = t_values[i + 1].item()
            t_batch = torch.full((shape[0],), t_curr, device=device).clamp(
                T_CLIP_MIN, T_CLIP_MAX
            )
            t_next_batch = torch.full((shape[0],), t_next, device=device).clamp(
                T_CLIP_MIN, T_CLIP_MAX
            )

            # Euler predictor step
            v1 = self._get_velocity(z, t_batch)
            z_predict = z + dt * v1

            # Corrector step (if not at last step)
            if i < n_steps - 1:
                v2 = self._get_velocity(z_predict, t_next_batch)
                z = z + dt * 0.5 * (v1 + v2)
            else:
                # Last step: just use Euler (no point correcting at the endpoint)
                z = z_predict

        return z


# =============================================================================
# Self-test  (run with: python -m src.flow_matching  from project root)
# =============================================================================

if __name__ == "__main__":
    import torch as _torch

    print("=== experiments.natural_images.common.flow_matching self-test ===\n")

    B, D = 32, 16
    _torch.manual_seed(0)

    x = _torch.randn(B, D)
    eps = _torch.randn(B, D)
    t = sample_logit_normal(B)
    t_b = _reshape_t(t, x)
    z_t = t_b * x + (1 - t_b) * eps

    # --- Time sampling distribution check ---
    t_large = sample_logit_normal(10000)
    assert t_large.min() >= T_CLIP_MIN and t_large.max() <= T_CLIP_MAX
    print(
        f"[PASS] sample_logit_normal: min={t_large.min():.4f} max={t_large.max():.4f} mean={t_large.mean():.4f}"
    )

    # --- Conversion roundtrips: x → eps → x ---
    x_from_eps = convert_pred_to_x(convert_pred_to_eps(x, z_t, t, "x"), z_t, t, "eps")
    assert _torch.allclose(x, x_from_eps, atol=1e-4), "x→eps→x roundtrip failed"
    print("[PASS] roundtrip x → eps → x")

    # --- Conversion roundtrips: x → v → x ---
    x_from_v = convert_pred_to_x(convert_pred_to_v(x, z_t, t, "x"), z_t, t, "v")
    assert _torch.allclose(x, x_from_v, atol=1e-4)
    print("[PASS] roundtrip x → v → x")

    # --- Conversion roundtrips: eps → v → eps ---
    eps_from_v = convert_pred_to_eps(convert_pred_to_v(eps, z_t, t, "eps"), z_t, t, "v")
    assert _torch.allclose(eps, eps_from_v, atol=1e-4)
    print("[PASS] roundtrip eps → v → eps")

    # --- Identity: convert x to x returns same tensor ---
    x_to_x = convert_pred_to_x(x, z_t, t, "x")
    assert x_to_x is x
    print("[PASS] identity conversion (x→x) returns same object")

    # --- v=x-eps definition: recover x from true v via z_t ---
    v_true = x - eps
    x_rec = convert_pred_to_x(v_true, z_t, t, "v")
    assert _torch.allclose(x_rec, x, atol=1e-4), "v=x-eps definition broken"
    print("[PASS] v=x-eps definition: recover x from true v")

    # --- FlowMatching wrapper: loss is a scalar > 0 for all 9 combos ---
    class _TinyNet(_torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = _torch.nn.Linear(D, D)

        def forward(self, z, t):
            return self.w(z)

    for ps in ["x", "eps", "v"]:
        for ls in ["x", "eps", "v"]:
            model = FlowMatching(_TinyNet(), pred_space=ps, loss_space=ls)
            loss = model.compute_loss(x)
            assert loss.ndim == 0 and loss.item() > 0, (
                f"Bad loss for pred={ps} loss={ls}"
            )
    print("[PASS] compute_loss is scalar > 0 for all 9 (pred_space, loss_space) combos")

    # --- Sampling: output shapes ---
    model_x = FlowMatching(_TinyNet(), pred_space="x", loss_space="v")
    s_euler = model_x.sample((4, D), n_steps=5, method="euler")
    assert s_euler.shape == (4, D)
    print(f"[PASS] Euler sample shape: {s_euler.shape}")

    s_heun = model_x.sample((4, D), n_steps=5, method="heun")
    assert s_heun.shape == (4, D)
    print(f"[PASS] Heun sample shape: {s_heun.shape}")

    print("\n=== All flow_matching tests passed ===")
