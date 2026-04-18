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

- `prototyping/`
  Transitional prototype material and fallback references. The cleaned
  `prototyping/gdfm_components.py` file remains the style and notation exemplar.

- `archives/`
  Zip snapshots of removed `experiments/` and `reference/` trees. Use these as
  historical context only.

- `baselines/` and `third_party/`
  External or vendored code. Treat as read-only unless explicitly asked.

## Placement Rules

- Put model-intrinsic logic in `src/gdfm/`.
- Put shared study infrastructure in the appropriate `src/mp_*` package.
- Keep fallback or transition code in `prototyping/`, not in canonical study
  code.
- If you need something from the archived trees, copy the minimum needed logic
  into `src/` and make it local there.

## GDFM Safety Rules

- Keep `prototyping/gdfm_components.py` intact as the reference fallback while
  canonical GDFM evolves in `src/gdfm/core.py`.
- Keep the prototype-vs-canonical parity tests green whenever canonical GDFM
  changes.
- Do not bake `prediction_space` or `loss_space` semantics into backbone
  networks. The network stays architecture-only.

## Artifact Rules

- Save the exact training data artifact for every run.
- Save the exact `z_init` used for checkpoint evaluation.
- Evaluate memorization against saved run artifacts, never against rebuilt data.
- Keep outputs local to the caller's chosen run directory.

## Environment

- Before running project-dependent Python commands, load and activate the
  project environment with `module load Mambaforge` and
  `mamba activate mem_param_env`.
- The bare system Python is not the project environment.

## Python Style

- Canonical first-party Python follows `PYTHON_STYLE.md`.
- If the guide conflicts with older code, the guide wins.
- Canonical first-party Python should not use type hints or
  `from __future__ import annotations`.

## Do Not

- Do not recreate a live dependency from `src/` into archived material.
- Do not create vague catch-all modules such as `utils.py` unless there is a
  very clear, narrow purpose.
- Do not edit vendored code under `baselines/*/vendor` or `third_party/`
  unless explicitly asked.
- Do not push to remote or amend past commits without explicit instruction.

## Workflow

1. Use `prototyping/` or the archives for context when needed.
2. Promote stable model or infrastructure code into `src/`.
3. Keep the safety suite green before trusting the canonical path.
4. After meaningful work, append or update `CODING_JOURNAL.md` with a UTC ISO
   timestamp, decisions, commands/checks, results, blockers, and next action.
