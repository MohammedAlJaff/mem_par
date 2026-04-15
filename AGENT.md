# AGENT Guide

Short operational guide for agents and contributors working in this repo.

## Canonical Surfaces

- `src/gdfm/`
  Canonical model-core package. Keep GDFM math, model setup, backbones, and
  future sampling/conversion helpers here.

- `src/mp_config/`, `src/mp_data/`, `src/mp_metrics/`, `src/mp_artifacts/`, `src/mp_training/`
  Thin reusable study infrastructure. These packages own config helpers, data
  loading, memorization metrics, artifact persistence, and the reusable
  train/evaluate skeleton.

- `experiments/<study>/`
  Canonical study surface. Each active study should stay thin and own
  `config.py`, `train.py`, `evaluate.py`, `plot.py`, plus local `runs/` and
  `figures/`.

- `experiments/legacy/`
  Preserved pre-reset experiment stack. Keep it readable, but do not treat it
  as the source of truth for new first-party work.

- `reference/`
  Non-canonical walkthroughs, sanity checks, exploratory notebooks, and older
  archived reference code.

- `prototyping/`
  Transitional prototype material and fallback references. The cleaned
  `prototyping/gdfm_components.py` file remains the style and notation exemplar.

- `baselines/` and `third_party/`
  External or vendored code. Treat as read-only unless explicitly asked.

## Placement Rules

- Put model-intrinsic logic in `src/gdfm/`.
- Put shared study infrastructure in the appropriate `src/mp_*` package.
- Put experiment-specific logic in the owning `experiments/<study>/` folder.
- Put pedagogy, sanity checks, and notebook exploration in `reference/`.
- Keep fallback or transition code in `prototyping/`, not in canonical study
  code.

## Canonical Experiment Contract

Every active study should be understandable from its own folder:

- `config.py` returns a `ConfigDict`
- `train.py` is the training entrypoint
- `evaluate.py` replays checkpoints and computes memorization metrics
- `plot.py` turns saved metrics/artifacts into figures
- `runs/` holds local run outputs
- `figures/` holds study-level figures

Active study code should reuse `src/` packages instead of rebuilding parallel
infrastructure.

## GDFM Safety Rules

- Keep `prototyping/gdfm_components.py` intact as the reference fallback while
  canonical GDFM evolves in `src/gdfm/core.py`.
- Keep the prototype-vs-canonical parity tests green whenever canonical GDFM
  changes.
- Do not bake `prediction_space` or `loss_space` semantics into backbone
  networks. The network stays architecture-only.

## Artifact Rules

- Save the exact training data artifact for every run.
- Save the exact `z_init` used for checkpoint replay.
- Evaluate memorization against saved run artifacts, never against rebuilt data.
- Keep outputs local to the owning experiment folder.

## Python Style

- Canonical first-party Python follows `PYTHON_STYLE.md`.
- If the guide conflicts with older code, the guide wins.
- Canonical first-party Python should not use type hints or
  `from __future__ import annotations`.

## Do Not

- Do not add new first-party experiment logic under `experiments/legacy/`.
- Do not import canonical runtime code from `reference/archive/`.
- Do not create vague catch-all modules such as `utils.py` unless there is a
  very clear, narrow purpose.
- Do not edit vendored code under `baselines/*/vendor` or `third_party/`
  unless explicitly asked.
- Do not push to remote or amend past commits without explicit instruction.

## Workflow

1. Explore or teach ideas in `reference/` or `prototyping/`.
2. Promote stable model or infrastructure code into `src/`.
3. Wire paper-facing study logic into `experiments/<study>/`.
4. Keep the safety suite green before trusting the new canonical path.
