"""
Diffusion and Flow Matching models.

Basic idea:
One codebase, that can accommodate both diffusion and flow matching models, by
swaping various schedules and regression target functions.

Idea:

interpolation and intermediate states from noise to sample


x(t) = a(t)x + b(t)z

and the generalized regression target can be viewd as

r(x, t, z) = c(t)x + d(t)z


"""

import torch

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
T_MU = -0.8  # HYPERPARAMETER: Li & He default
T_SIGMA = 0.8  # HYPERPARAMETER: Li & He default
# μ = 0, σ = 1 gives more uniform coverage
# μ = -1.2, σ = 0.8 upweights low-noise regime (helps x-pred)

# =============================================================================
# Time sampling
# =============================================================================


def sample_logit_normal(
    batch_size: int,
    mu: float = T_MU,
    sigma: float = T_SIGMA,
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


def sample_uniform_time(batch_size: int, device: str = "cpu") -> torch.Tensor:
    """
    Sample t from a uniform distribution.

    Returns:
        (B,) tensor of timesteps in [T_CLIP_MIN, T_CLIP_MAX]
    """
    t = torch.rand(batch_size, device=device)
    t = t.clamp(T_CLIP_MIN, T_CLIP_MAX)
    return t


if __name__ == "__main__":
    ###############
    #### Test time sampling ####
    """
    What this does: 
    - Samples N = batch_size samples from a logit-normal distribution
    - Samples N = batch_size samples from a uniform distribution
    - Plots histograms of both distributions

    What to expect: 
    - Logit-normal samples should be concentrated around 0.31
    - Uniform samples should be uniformly distributed
    """
    batch_size = 100000
    device = "cpu"
    t_uniform = sample_uniform_time(batch_size, device=device)
    print("Uniform samples:", t_uniform)

    t_logit_normal = sample_logit_normal(batch_size, device=device)
    print("Logit-normal samples:", t_logit_normal)

    # hist
    import matplotlib.pyplot as plt

    _ = plt.hist(t_uniform.cpu().numpy(), bins=50, alpha=0.5, label="Uniform")
    _ = plt.hist(t_logit_normal.cpu().numpy(), bins=50, alpha=0.5, label="Logit-normal")
    plt.legend()
    plt.show()
