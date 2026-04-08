"""
Lab 10 - Generalized denoiser prototype.

This file is meant to be read slowly.

It is the smallest self-contained prototype of the paper's training-side
reformulation. The point is not to be "production ready" yet. The point is to
make the abstraction visible and easy to reason about:

    z = a(t) * x + b(t) * eps
    r(x, eps, t) = c(t) * x + d(t) * eps
    L = E[ w(t) * ||NN(z, t) - r||^2 ]

The key idea is that many denoising generative families can be described by the
same training template if we change only the schedule functions:

    - a(t), b(t): how we corrupt clean data x into noisy input z
    - c(t), d(t): what regression target NN should predict
    - w(t): how strongly each time/noise level is weighted in the loss

In other words, the "family" lives in the math schedules, not in a totally
different training loop.

For the paper's experiment design, we also want target choice to be separate
from path choice. So in this version we split the setup into:

    - a path spec: chooses a(t), b(t), p(t), and w(t)
    - a representation spec: chooses c(t), d(t)

This makes representations such as x, eps, and v orthogonal to the choice of
path family such as FM, DDPM, or EDM.

This first script is intentionally minimal:
    - one file
    - no dependency on the repo's current FlowMatching code
    - path presets described by schedules rather than subclasses
    - one tiny time-conditioned MLP used only as a dummy stand-in for NN

Just as important: the objective is NN-agnostic. Nothing below assumes
U-Net, Transformer, MLP, or any other specific architecture. The only contract
is:

    NN(z, t) -> prediction

This first version only covers Eq. (1)-(3), i.e. the training formulation.
Sampling from Eq. (4) is deliberately left for the next prototype so we can get
the training abstraction clean first.

Run from project root:
    python lab/10_generalized_denoiser_prototype.py
"""

from __future__ import annotations  # enables forward references in type hints

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


# Short aliases keep the rest of the file readable.
# They are not conceptually deep; they just reduce visual clutter.
Tensor = torch.Tensor

# A "schedule function" takes a batch of time values and returns a batch of
# coefficients such as a(t), b(t), c(t), d(t), or w(t).
ScheduleFn = Callable[[Tensor], Tensor]

# A "time sampler" defines p(t): how we choose time/noise levels during
# training. Different families may prefer different sampling distributions.
TimeSampler = Callable[[int, torch.device], Tensor]

# A representation chooses the target coefficients c(t), d(t).
# It depends on the chosen path because the meaning of "v" is path-dependent.
RepresentationCoeffFn = Callable[["PathSpec", Tensor], tuple[Tensor, Tensor]]


def seed_everything(seed: int) -> None:
    """Seed the small number of RNGs used in this toy script.

    For a prototype, reproducibility is extremely helpful. If a sanity check
    behaves differently every run, it becomes much harder to tell whether a
    change in code actually helped or hurt.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _expand_like(values: Tensor, ref: Tensor) -> Tensor:
    """Broadcast a 1D batch of coefficients so it can multiply a data tensor.

    Example:
        values has shape [batch]
        ref has shape [batch, dim]

    We want to turn values into shape [batch, 1] so that PyTorch can broadcast
    it across all feature dimensions of ref.
    """

    while values.dim() < ref.dim():
        values = values.unsqueeze(-1)
    return values


def _mean_squared_error_per_example(pred: Tensor, target: Tensor) -> Tensor:
    """Return one scalar MSE for each example in the batch.

    We flatten all non-batch dimensions and average within each sample. This is
    useful because Eq. (2) applies the weight w(t) per training example, so we
    want the per-example error before averaging across the batch.
    """

    return (pred - target).reshape(pred.shape[0], -1).pow(2).mean(dim=1)


def _sample_uniform(
    batch_size: int, device: torch.device, t_min: float, t_max: float
) -> Tensor:
    """Sample time values uniformly from [t_min, t_max].

    This is the simplest possible choice for p(t). It treats all parts of the
    time interval equally.
    """

    t = torch.rand(batch_size, device=device)
    return t_min + (t_max - t_min) * t


def _sample_log_uniform(
    batch_size: int, device: torch.device, t_min: float, t_max: float
) -> Tensor:
    """Sample time values log-uniformly between t_min and t_max.

    Log-uniform sampling is useful when the interesting scale is multiplicative
    rather than additive. EDM-style sigma ranges are a classic example.
    """

    log_min = math.log(t_min)
    log_max = math.log(t_max)
    u = torch.rand(batch_size, device=device)
    return torch.exp(log_min + (log_max - log_min) * u)


def _cosine_alpha_bar(t: Tensor, s: float = 0.008) -> Tensor:
    """Cosine noise schedule used here as a lightweight DDPM-style example.

    alpha_bar(t) is the cumulative signal preservation term. In diffusion
    language, sqrt(alpha_bar) multiplies the clean sample and
    sqrt(1 - alpha_bar) multiplies the noise.

    The clamp avoids numerical issues exactly at the endpoints.
    """

    shifted = (t + s) / (1.0 + s)
    alpha_bar = torch.cos(0.5 * math.pi * shifted).pow(2)
    alpha_bar_0 = math.cos(0.5 * math.pi * (s / (1.0 + s))) ** 2
    return (alpha_bar / alpha_bar_0).clamp(1.0e-5, 1.0 - 1.0e-5)


@dataclass
class PathSpec:
    """Container for one path/family specification.

    Think of this as the part of the training setup that defines how noisy
    states are formed and how time is sampled.

    Fields:
        name:
            Human-readable identifier used in printouts.
        t_min, t_max:
            Valid range of time/noise values used in training.
        sample_t:
            Implementation of p(t), i.e. how t is sampled during training.
        a, b:
            Corruption schedules in z = a(t) x + b(t) eps.
        w:
            Loss weight schedule in Eq. (2).
        v_definition:
            Which path-specific meaning of "v" should be used.
            For now we support:
                - "fm": native flow-matching velocity target
                - "vp": diffusion-style v target with coefficients (-b, a)
        summary_t:
            A few representative time values used only for readable summaries.
    """

    name: str
    t_min: float
    t_max: float
    sample_t: TimeSampler
    a: ScheduleFn
    b: ScheduleFn
    w: ScheduleFn
    v_definition: str
    summary_t: tuple[float, float, float]


@dataclass
class RepresentationSpec:
    """Container for one target representation.

    This is the object that tells us what the direct output of NN should mean.
    In the paper's language, this is the prediction-space choice.

    The same type of object can later be reused for loss space as well.
    """

    name: str
    coeffs: RepresentationCoeffFn


def make_path_presets() -> dict[str, PathSpec]:
    """Build the first small collection of path presets.

    These are path choices only. They no longer hard-code the prediction target.
    """

    def ones(t: Tensor) -> Tensor:
        """A simple default weighting function w(t) = 1."""

        return torch.ones_like(t)

    # Flow Matching in the paper's reformulation.
    #
    # z = (1 - t) x + t eps
    #
    # For this path, the native FM velocity target in our current convention is
    # v = eps - x, i.e. coefficient pair (-1, 1).
    fm = PathSpec(
        name="fm",
        t_min=0.0,
        t_max=1.0,
        sample_t=lambda batch_size, device: _sample_uniform(
            batch_size, device, 0.0, 1.0
        ),
        a=lambda t: 1.0 - t,
        b=lambda t: t,
        w=ones,
        v_definition="fm",
        summary_t=(0.0, 0.5, 1.0),
    )

    # A simple DDPM-style path written in the same unified language.
    ddpm = PathSpec(
        name="ddpm",
        t_min=0.0,
        t_max=1.0,
        sample_t=lambda batch_size, device: _sample_uniform(
            batch_size, device, 0.0, 1.0
        ),
        a=lambda t: torch.sqrt(_cosine_alpha_bar(t)),
        b=lambda t: torch.sqrt(1.0 - _cosine_alpha_bar(t)),
        w=ones,
        v_definition="vp",
        summary_t=(0.0, 0.5, 1.0),
    )

    # sigma_data is the fixed data-scale hyperparameter used by EDM.
    # We keep it local to this preset because, in this script, the goal is to
    # show the schedule form rather than expose every knob as configuration.
    sigma_data = 0.5

    def edm_norm(t: Tensor) -> Tensor:
        """Common denominator that appears repeatedly in the EDM schedules."""

        return torch.sqrt(t.pow(2) + sigma_data**2)

    # EDM written in terms of the raw network output, matching the paper's
    # reformulation rather than the wrapped/preconditioned presentation.
    edm = PathSpec(
        name="edm",
        t_min=0.02,
        t_max=1.50,
        sample_t=lambda batch_size, device: _sample_log_uniform(
            batch_size, device, 0.02, 1.50
        ),
        a=lambda t: t / edm_norm(t),
        b=lambda t: torch.full_like(t, sigma_data) / edm_norm(t),
        w=ones,
        v_definition="vp",
        summary_t=(0.02, math.sqrt(0.02 * 1.50), 1.50),
    )

    return {spec.name: spec for spec in (fm, ddpm, edm)}


def make_representation_presets() -> dict[str, RepresentationSpec]:
    """Build the small set of target representations used in the paper.

    Important note:
    The meaning of "v" depends on the path family. That is exactly why this
    function returns coefficients as a function of both the path and time.
    """

    def x_coeffs(path: PathSpec, t: Tensor) -> tuple[Tensor, Tensor]:
        del path
        return torch.ones_like(t), torch.zeros_like(t)

    def eps_coeffs(path: PathSpec, t: Tensor) -> tuple[Tensor, Tensor]:
        del path
        return torch.zeros_like(t), torch.ones_like(t)

    def v_coeffs(path: PathSpec, t: Tensor) -> tuple[Tensor, Tensor]:
        if path.v_definition == "fm":
            # For the FM path used in this script, the native velocity target is
            # eps - x, so (c, d) = (-1, 1).
            return -torch.ones_like(t), torch.ones_like(t)
        if path.v_definition == "vp":
            # Diffusion-style v is a rotation of (x, eps), usually written with
            # coefficients (-sigma_t, alpha_t). In our generic notation this is
            # simply (-b(t), a(t)).
            return -path.b(t), path.a(t)
        raise ValueError(f"Unsupported v definition: {path.v_definition}")

    return {
        "x": RepresentationSpec(name="x", coeffs=x_coeffs),
        "eps": RepresentationSpec(name="eps", coeffs=eps_coeffs),
        "v": RepresentationSpec(name="v", coeffs=v_coeffs),
    }


class ScalarTimeEmbedding(nn.Module):
    """Map a scalar time value t into a richer feature vector.

    Why do this at all?
    A raw scalar can be hard for an NN to use directly. Sinusoidal
    features let the NN see time at multiple frequencies/scales, which is
    a standard trick in sequence models and diffusion-style models.

    In this first prototype we keep only the conditioned case, so the time
    embedding is always part of the network input.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        # We create half sine channels and half cosine channels.
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half, 1)
        )

        # Each time value is evaluated at all frequencies.
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        out = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        # If dim is odd, pad by one feature so the output dimension matches the
        # requested size exactly.
        if self.dim % 2:
            out = torch.cat([out, torch.zeros_like(t.unsqueeze(1))], dim=1)
        return out


class DummyTimeConditionedNN(nn.Module):
    """A tiny MLP that explicitly receives time information.

    This is not meant to be a good generative model. It is just a pedagogical
    placeholder showing how the generalized objective can work with an NN that
    uses explicit time/noise conditioning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        time_embed_dim: int = 32,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            ScalarTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        # NN sees both z and the learned time features.
        layers: list[nn.Module] = []
        current_dim = input_dim + time_embed_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        # Convert scalar t into a vector embedding, then concatenate it with z.
        t_features = self.time_embed(t)
        inputs = torch.cat([z, t_features], dim=1)
        return self.net(inputs)


class GeneralizedDenoisingObjective:
    """
    Training-only prototype for the reformulation in Eq. (1)-(3).

    The objective does not know anything about the NN architecture.
    It only assumes the NN implements:

        NN(z, t) -> prediction
    """

    def __init__(self, path: PathSpec, representation: RepresentationSpec):
        self.path = path
        self.representation = representation

    def sample_batch(self, x: Tensor) -> dict[str, Tensor]:
        """Construct one generalized denoising training batch.

        Starting from clean data x, we:
            1. sample Gaussian noise eps
            2. sample time values t ~ p(t)
            3. build noisy inputs z using a(t), b(t)
            4. build regression targets using the chosen representation
            5. compute per-example loss weights w(t)

        The result is a dictionary so that later debugging and sanity checks are
        easy: we can inspect every intermediate quantity directly.
        """

        batch_size = x.shape[0]
        device = x.device

        # eps plays the role of the random corruption/noise variable.
        eps = torch.randn_like(x)

        # The path decides how time values are sampled.
        t = self.path.sample_t(batch_size, device)

        # The schedules are defined over 1D time values, so we expand them to
        # make them broadcast correctly over vector-valued data x.
        a_t = _expand_like(self.path.a(t), x)
        b_t = _expand_like(self.path.b(t), x)

        # The representation decides what NN is asked to predict.
        c_raw, d_raw = self.representation.coeffs(self.path, t)
        c_t = _expand_like(c_raw, x)
        d_t = _expand_like(d_raw, x)

        # Eq. (1): noisy observation / latent / corrupted input.
        z = a_t * x + b_t * eps

        # Eq. (3): regression target corresponding to this representation.
        target = c_t * x + d_t * eps

        # Eq. (2): per-example weighting before batch averaging.
        weight = self.path.w(t)

        return {
            "x": x,
            "eps": eps,
            "t": t,
            "z": z,
            "target": target,
            "weight": weight,
        }

    def compute_loss(self, NN: nn.Module, x: Tensor) -> dict[str, Tensor]:
        """Run the full generalized loss computation for one mini-batch."""

        batch = self.sample_batch(x)

        # The objective does not care about the NN architecture itself.
        # It only relies on the simple contract: give the NN (z, t) and
        # receive a prediction in the chosen representation.
        pred = NN(batch["z"], batch["t"])

        # First compute one MSE number per training example.
        per_example_mse = _mean_squared_error_per_example(pred, batch["target"])

        # Then apply w(t) and average across the batch, matching Eq. (2).
        loss = (batch["weight"] * per_example_mse).mean()
        batch["pred"] = pred
        batch["loss"] = loss
        return batch


def sample_toy_data(batch_size: int, input_dim: int, device: torch.device) -> Tensor:
    """Generate a tiny synthetic dataset for training sanity checks.

    We use a two-cluster Gaussian mixture. This is intentionally simple: the
    goal is not to benchmark model quality, only to make sure the objective and
    optimization code behave sensibly.
    """

    # Two opposite cluster centers give us a very easy toy distribution.
    centers = torch.stack(
        [
            torch.ones(input_dim, device=device),
            -torch.ones(input_dim, device=device),
        ],
        dim=0,
    )

    # Each sample chooses which cluster it came from.
    assignments = torch.randint(0, centers.shape[0], (batch_size,), device=device)

    # Add a little Gaussian spread around each center.
    noise = 0.30 * torch.randn(batch_size, input_dim, device=device)
    return centers[assignments] + noise


def summarize_setup(path: PathSpec, representation: RepresentationSpec) -> None:
    """Print a small table of schedule values for one path/representation pair.

    This is one of the fastest sanity checks when building a new setup:
    before training anything, make sure the coefficients "look right."
    """

    print(f"\n--- Setup: path={path.name}, rep={representation.name} ---")
    t = torch.tensor(path.summary_t, dtype=torch.float32)
    a = path.a(t)
    b = path.b(t)
    c, d = representation.coeffs(path, t)
    w = path.w(t)

    print("  t       a(t)     b(t)     c(t)     d(t)     w(t)")
    for idx in range(t.numel()):
        print(
            "  "
            f"{t[idx].item():>5.3f}   "
            f"{a[idx].item():>7.4f}   "
            f"{b[idx].item():>7.4f}   "
            f"{c[idx].item():>7.4f}   "
            f"{d[idx].item():>7.4f}   "
            f"{w[idx].item():>7.4f}"
        )


def schedule_sanity_check(
    path: PathSpec,
    representation: RepresentationSpec,
    input_dim: int,
    device: torch.device,
) -> None:
    """Run a few cheap checks that catch common implementation mistakes.

    These are not proofs of correctness. They are just the kind of guardrails
    that help during early prototyping:
        - shapes should line up
        - values should be finite
        - known endpoint identities should hold for FM
    """

    objective = GeneralizedDenoisingObjective(path, representation)
    x = sample_toy_data(batch_size=8, input_dim=input_dim, device=device)
    batch = objective.sample_batch(x)

    # Basic tensor-shape expectations.
    assert batch["z"].shape == x.shape
    assert batch["target"].shape == x.shape
    assert batch["weight"].shape == (x.shape[0],)

    # Numerical explosions or NaNs this early usually mean a schedule bug.
    assert torch.isfinite(batch["z"]).all()
    assert torch.isfinite(batch["target"]).all()
    assert torch.isfinite(batch["weight"]).all()

    if path.name == "fm":
        # FM has especially simple endpoints, so it is a nice path for exact
        # identity checks. At t=0 we should recover pure data, and at t=1 we
        # should recover pure noise.
        t0 = torch.tensor([0.0], device=device)
        t1 = torch.tensor([1.0], device=device)
        assert torch.allclose(path.a(t0), torch.ones_like(t0))
        assert torch.allclose(path.b(t0), torch.zeros_like(t0))
        assert torch.allclose(path.a(t1), torch.zeros_like(t1))
        assert torch.allclose(path.b(t1), torch.ones_like(t1))

    # The x and eps targets are exact, path-independent coefficient pairs.
    t_probe = torch.tensor(path.summary_t, dtype=torch.float32, device=device)
    c_probe, d_probe = representation.coeffs(path, t_probe)
    if representation.name == "x":
        assert torch.allclose(c_probe, torch.ones_like(t_probe))
        assert torch.allclose(d_probe, torch.zeros_like(t_probe))
    if representation.name == "eps":
        assert torch.allclose(c_probe, torch.zeros_like(t_probe))
        assert torch.allclose(d_probe, torch.ones_like(t_probe))
    if representation.name == "v" and path.v_definition == "fm":
        assert torch.allclose(c_probe, -torch.ones_like(t_probe))
        assert torch.allclose(d_probe, torch.ones_like(t_probe))
    if representation.name == "v" and path.v_definition == "vp":
        assert torch.allclose(c_probe, -path.b(t_probe))
        assert torch.allclose(d_probe, path.a(t_probe))

    print(
        f"  [pass] setup sanity: path={path.name}, rep={representation.name}, "
        f"z shape={tuple(batch['z'].shape)}"
    )


def training_sanity_check(
    path: PathSpec,
    representation: RepresentationSpec,
    input_dim: int,
    device: torch.device,
    steps: int = 60,
) -> None:
    """Train for a few steps and check that the loss behaves sanely.

    This is intentionally weak as a test: we are not trying to prove convergence
    or compare methods. We only want to catch obvious failures such as:
        - the loss becoming NaN
        - gradients exploding immediately
        - a bug that makes optimization move in the wrong direction
    """

    objective = GeneralizedDenoisingObjective(path, representation)
    NN: nn.Module = DummyTimeConditionedNN(input_dim=input_dim).to(device)

    # Adam is more than enough for this toy prototype.
    optimizer = torch.optim.Adam(NN.parameters(), lr=1.0e-3)

    losses: list[float] = []
    for _ in range(steps):
        # Fresh toy data each step keeps the check simple and stochastic.
        x = sample_toy_data(batch_size=128, input_dim=input_dim, device=device)
        out = objective.compute_loss(NN, x)
        loss = out["loss"]

        # Standard PyTorch training step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # We compare the average of the early losses with the average of the late
    # losses. Averaging smooths out the noise of stochastic optimization.
    start_mean = sum(losses[:10]) / 10.0
    end_mean = sum(losses[-10:]) / 10.0
    delta = end_mean - start_mean
    print(
        f"  [train-check] {path.name:>4} | {representation.name:>4} | "
        f"start={start_mean:.5f} end={end_mean:.5f} delta={delta:+.5f}"
    )

    assert math.isfinite(start_mean)
    assert math.isfinite(end_mean)
    assert end_mean < start_mean * 1.20, "loss exploded during training sanity check"


def main() -> None:
    """Run the prototype as a script.

    The output is intentionally verbose and structured like a mini lab notebook:
        1. show path/representation schedules
        2. run cheap setup sanity checks
        3. run short training sanity checks
    """

    print("=" * 72)
    print("Lab 10: Generalized denoiser prototype")
    print("=" * 72)
    print("This script focuses on the training-side reformulation only.")
    print("Paths: fm, ddpm, edm")
    print("Representations: x, eps, v")
    print("NN: one tiny time-conditioned dummy MLP")

    seed_everything(0)

    # Use GPU if it is available, otherwise CPU. Nothing in this script depends
    # on GPU-specific behavior.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 16

    # Build the small set of path and representation presets we want to inspect.
    path_presets = make_path_presets()
    representation_presets = make_representation_presets()

    print(f"\nDevice: {device}")
    print(f"Input dimension: {input_dim}")
    print("\n1) Path/representation summaries")
    for path in path_presets.values():
        for representation in representation_presets.values():
            summarize_setup(path, representation)

    print("\n2) Setup sanity checks")
    for path in path_presets.values():
        for representation in representation_presets.values():
            schedule_sanity_check(
                path,
                representation,
                input_dim=input_dim,
                device=device,
            )

    print("\n3) Training sanity checks")
    for path in path_presets.values():
        for representation in representation_presets.values():
            training_sanity_check(
                path,
                representation,
                input_dim=input_dim,
                device=device,
            )

    print("\nDone. Next step: add the generic first-order sampler from Eq. (4).")


if __name__ == "__main__":
    main()
