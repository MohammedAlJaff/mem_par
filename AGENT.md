# AGENT Guide

Short operational guide for agents and contributors working in this repo.

## Communication Rules

- The user is a researcher. Use simple wording and precise research
  terminology.
- Do not use slang, hype wording, playful labels, or informal phrases.
- Prefer direct descriptions of methods, assumptions, metrics, results,
  blockers, and next actions.
- Keep responses short by default. Answer the question first, usually in one
  to three sentences. Add detail only when the user asks for it or when the
  next action would otherwise be ambiguous.
- Do not be long-winded. Prefer short bullets over long explanations.
- Examples:
  - Avoid: "Got it - let's clean this up."
    Use: "Understood. I will remove the unused metric fields and update the call sites."
  - Avoid: "This is a nice little smoke test."
    Use: "This is a short runtime test that checks whether the path executes without errors."
  - Avoid: "The code has some weird leftovers."
    Use: "The code still contains stale compatibility aliases from the previous metric interface."
  - Avoid: "This metric is doing oracle shit."
    Use: "This metric uses the known synthetic data-generating distribution."
  - Avoid: "The runner wires everything together."
    Use: "The runner constructs the config, launches training, and optionally runs checkpoint evaluation."
  - Avoid: "This is probably cosmetic test coverage."
    Use: "This test checks structure or naming, but does not verify behavioral correctness."
  - Avoid: "We can hard prune this."
    Use: "We can remove this interface without preserving backward compatibility."
  - Avoid: "The old names are noisy."
    Use: "The old names are ambiguous because they do not state the measured quantity or comparison set."
  - Avoid: "The figure plumbing is now aligned."
    Use: "The figure filenames, figure map keys, run summary entries, and W&B keys now use the same names."

## Canonical Surfaces

- `src/gdfm/`
  Canonical model-core package. Keep GDFM math, model setup, backbones, and
  future sampling/conversion helpers here.

- `src/mp_config/`, `src/mp_data/`, `src/mp_metrics/`, `src/mp_artifacts/`, `src/mp_training/`
  Thin reusable study infrastructure. These packages own config helpers, data
  loading, metric computation, artifact persistence, and the reusable training
  skeleton.

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
- Keep `mp_metrics` limited to metric computation from in-memory tensors or
  feature matrices. Put checkpoint replay in `mp_runners`, figure writing in
  `mp_visualization`, file/artifact layout in `mp_artifacts`, and tracker
  payload formatting in `mp_tracking`.
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
- When W&B is enabled and no explicit W&B flag is passed, derive project, run
  name, and group from the resolved config. Project is dataset+model. Run name
  includes dataset, model, `N`, batch cap, prediction/loss spaces, training
  steps, and seed. Group is dataset+model+training steps so `x`, `eps`, `v`,
  batch-size, seed, and `N` sweeps stay comparable.

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
- Use math-first names from the paper when the paper defines the quantity:
  `rho`, `r`, `d1`, `d2`, `N`, `tau`, and `t`.
- Use canonical metric keys in return dictionaries, JSON, and W&B logs. Do not
  add duplicate aliases such as `nn_ratio`, `ratios`, or `ratio_threshold` in
  memorization metric records.
- Inline comments for named quantities should state the formula or definition,
  for example `r = d1 / d2` or `delta_f_mem = f_mem(gen) - f_mem(held-out)`.

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
