# Handoff: Generalized Denoiser Prototype

This note is for a stronger follow-on model or future contributor who needs to
continue the work around the prototype in
[lab/10_generalized_denoiser_prototype.py](10_generalized_denoiser_prototype.py).

## Intent

The long-term goal is not merely to build a generic diffusion sandbox. The code
is meant to become an experiment engine for the paper draft currently titled:

> "Equivalent in Principle, Unequal in Practice:
> How Target Parameterization Shapes Memorization and Generalization in
> Diffusion and Flow-Based Models"

The central research question is:

- when model family, optimizer, and training protocol are held fixed, how do
  different target parameterizations change memorization dynamics and
  generalization under finite data and finite model capacity?

The code should therefore support controlled comparisons across:

- path family
- prediction space
- loss space
- network backbone
- dataset geometry
- memorization diagnostics over training

The intent is to make these axes explicit and separable rather than burying
them inside framework-specific classes.

## Current State

The prototype currently lives in one intentionally pedagogical script:

- [lab/10_generalized_denoiser_prototype.py](10_generalized_denoiser_prototype.py)

What it already does:

- implements the training-side reformulation
  - `z = a(t) x + b(t) eps`
  - `r(x, eps, t) = c(t) x + d(t) eps`
  - `L = E[w(t) ||NN(z, t) - r||^2]`
- separates `path` from `representation`
- uses `NN` terminology to align with the paper
- includes path presets for:
  - `fm`
  - `ddpm`
  - `edm`
- includes representation presets for:
  - `x`
  - `eps`
  - `v`
- uses one dummy time-conditioned MLP NN
- includes:
  - printed setup summaries
  - setup sanity checks
  - short training sanity checks

What it does not do yet:

- separate prediction space from loss space
- perform representation conversion between predicted and loss spaces
- implement sampling / inference
- integrate canonical repo experiment runners
- implement memorization metrics / checkpoint evaluation
- run in this environment end-to-end, because `torch` is not installed here

## Design Philosophy

These are the main non-negotiables that should survive future refactors.

### 1. Math-first, not framework-first

The architecture should be driven by the reformulation itself, not by named
methods as subclasses. `FM`, `DDPM`, and `EDM` should mostly be presets over a
shared mathematical interface.

### 2. Path choice and representation choice are different things

The code should continue to distinguish:

- path choice: how noisy states are formed and how time is sampled
- representation choice: what the NN is asked to predict

This distinction is central to the paper. It is the bridge from "equivalent in
principle" to "possibly unequal in practice."

### 3. Prediction space and loss space should become separate objects

Right now the prototype has only one `RepresentationSpec`, used as prediction
space. The next abstraction should use the same kind of object twice:

- `prediction_space`
- `loss_space`

That is how the paper's full `3 x 3` design should be expressed.

### 4. Keep the NN abstracted away from the objective

The objective should remain NN-agnostic. The contract should stay minimal:

- `NN(z, t) -> prediction`

The objective should not care whether the NN is an MLP, U-Net, JiT/ViT, or
something else.

### 5. Avoid ambiguous names, especially for `v`

The symbol `v` is overloaded in the literature. Do not rely on the bare string
`"v"` as if it meant the same thing everywhere.

At minimum, the implementation must preserve path-aware `v` semantics, and a
fuller implementation may want explicit names such as:

- `v_fm`
- `v_vp`

### 6. Keep the code readable for junior researchers

This prototype is intentionally comment-heavy and pedagogical. That is not
accidental. Future refactors should preserve this readability, especially in
the core math layers.

## Core Abstractions Already Present

### `PathSpec`

Represents the path family. It currently owns:

- `a(t)`
- `b(t)`
- `p(t)` through `sample_t`
- `w(t)`
- a `v_definition` field for path-specific `v` meaning

Conceptually, this is the family/path layer.

### `RepresentationSpec`

Represents what the direct NN output means. It currently returns coefficients
`c(t), d(t)` as a function of both path and time.

Conceptually, this is the prediction-space layer.

### `GeneralizedDenoisingObjective`

Owns the shared training logic:

- sample `eps`
- sample `t ~ p(t)`
- build `z`
- build target coefficients from representation choice
- compute weighted MSE

It is intentionally ignorant of NN architecture details.

## Current Mathematical Convention

The current prototype follows the reformulation:

- `z = a(t) x + b(t) eps`
- `r = c(t) x + d(t) eps`

Current path presets:

- `fm`
  - `a(t) = 1 - t`
  - `b(t) = t`
- `ddpm`
  - `a(t) = sqrt(alpha_bar(t))`
  - `b(t) = sqrt(1 - alpha_bar(t))`
- `edm`
  - uses the raw-network reformulation rather than EDM's wrapped output

Current representation presets:

- `x`
  - `(c, d) = (1, 0)`
- `eps`
  - `(c, d) = (0, 1)`
- `v`
  - for `fm`, uses the current native FM convention:
    - `(c, d) = (-1, 1)` which means `eps - x`
  - for VP-style paths (`ddpm`, `edm`), uses:
    - `(c, d) = (-b(t), a(t))`

## Important Convention Warning

The FM sign/path convention is a likely future source of confusion.

The prototype currently uses the convention from the reformulation table:

- `z = (1 - t) x + t eps`
- `v = eps - x`

However, the paper draft also contains a later flow-matching section with the
opposite interpolation direction:

- `z_t = t x + (1 - t) eps`
- velocity `x - eps`

These are related by time reversal, but the implementation must never pretend
they are identical without making the convention explicit.

Recommendation:

- always encode the path explicitly
- never rely on a bare `v` label without knowing which path convention is active

## Immediate Next Step

The next important refactor is not sampling yet. It is the separation of
prediction space from loss space.

Introduce a training spec shaped like:

- `path`
- `prediction_space`
- `loss_space`
- `weight`
- `NN`

The key missing operation is representation conversion.

If `NN` predicts in one representation and loss is measured in another, we need
to recover `x_hat` and `eps_hat` from `(z, y_pred)` and then express the result
in the loss representation.

The generic conversion logic should come from solving the linear system:

- `z = a x + b eps`
- `y_from = c1 x + d1 eps`

Then:

- `det = a d1 - b c1`
- `x_hat = (d1 z - b y_from) / det`
- `eps_hat = (-c1 z + a y_from) / det`
- `y_to = c2 x_hat + d2 eps_hat`

This conversion layer is the heart of the future `3 x 3` grid.

## After That

Once prediction/loss separation exists, the next steps should be:

1. Add a `LossSpaceSpec` or simply reuse `RepresentationSpec` for loss space.
2. Add explicit off-diagonal training.
3. Add a `SamplerSpec` for Eq. (4)-style first-order sampling.
4. Add dataset wrappers and evaluation hooks.
5. Add memorization tracking over checkpoints.
6. Promote the abstraction from `lab/` into the experiment stack only after the
   math interface feels stable.

## Evaluation Requirements From The Paper

The final system should support, or cleanly grow into support for:

- memorization fraction based on nearest-neighbor ratio
- nearest-neighbor distance and ratio over training
- `tau_mem` style memorization-onset summaries
- synthetic geometry sweeps over:
  - ambient dimension `D`
  - intrinsic dimension `d`
  - dataset size `N`
  - ratio `D / d`
- image experiments with fixed subsamples across parameterization settings
- consistent seed handling across compared settings

This is important: the code is not just for producing samples. It is for making
controlled scientific comparisons.

## Reference Papers and Threads

These are the most relevant references for the current direction.

### Direct formulation anchor

- Sun, Jiang, Zhao, He:
  - "Is Noise Conditioning Necessary for Denoising Generative Models?"
  - This provides the reformulation:
    - `z = a(t) x + b(t) eps`
    - `r = c(t) x + d(t) eps`
    - unified first-order sampler notation
  - It is the main mathematical template for the prototype.

### Family references

- Ho et al.:
  - `DDPM`
- Karras et al.:
  - `EDM`
- Lipman et al.:
  - `Flow Matching`

### Parameterization motivation

- Li et al.:
  - "Back to Basics"
  - Main motivation for the idea that parameterization matters under finite
    capacity, especially in high-codimension regimes.

### Memorization / generalization references in the paper draft

- Gu et al.
- Yoon et al.
- Zhang et al.
- Bonnaire et al.
- George et al.

These motivate the memorization framing, the data/capacity framing, and the
random-feature surrogate lens in the theory section.

## Things To Preserve

- keep `NN` naming aligned with the paper
- keep path choice separate from representation choice
- keep representation logic explicit in coefficient form
- keep comments pedagogical
- keep the prototype in one script until the abstractions stabilize
- do not prematurely bind the objective to an MLP/U-Net/JiT-specific interface

## Suggested Prompt For The Next Model

If handing this to a stronger model, a good prompt would be:

> Continue from
> [lab/10_generalized_denoiser_prototype.py](10_generalized_denoiser_prototype.py)
> and
> [lab/10_generalized_denoiser_handoff.md](10_generalized_denoiser_handoff.md).
> Preserve the current math-first design. Do not collapse the code into
> framework-specific subclasses. Keep `NN` naming aligned with the paper.
> The immediate task is to add a clean separation between prediction space and
> loss space using generic representation conversion, while keeping the code
> pedagogical and the NN abstraction intact. Be careful with FM `v` sign/path
> conventions and do not use a bare ambiguous `v` internally if a more explicit
> name is needed.
