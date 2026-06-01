# Agent Guide

Canonical instructions for agents and contributors working in this repo.

## Project Framing

Paper: "Equivalent in Principle, Unequal in Practice"

Current repo question:
- Under fixed architecture, data, optimizer, sampler, and evaluation protocol,
  how do target parameterizations (`x`, `eps`, `v`) change memorization
  dynamics?

Conference framing:
- Target parameterization changes the route from generation to memorization.

Key measurements:
- `f_mem`
- nearest-neighbor ratio and distance statistics
- checkpoint-level memorization trajectories over training

## Communication Rules

- The user is a researcher. Use simple wording and precise research
  terminology.
- Do not use slang, hype wording, playful labels, or informal phrases.
- Prefer direct descriptions of methods, assumptions, metrics, results,
  blockers, and next actions.
- Keep responses short by default. Answer the question first, usually in one
  to three sentences. Add detail only when the next action would otherwise be
  ambiguous.
- Prefer short bullets over long explanations.

Examples:
- Avoid: "This is a nice little smoke test."
  Use: "This is a short runtime test that checks whether the path executes
  without errors."
- Avoid: "This metric is doing oracle shit."
  Use: "This metric uses the known synthetic data-generating distribution."
- Avoid: "The runner wires everything together."
  Use: "The runner constructs the config, launches training, and optionally
  runs checkpoint evaluation."

## Engineering Discipline

- Think before coding. State assumptions, expose ambiguity, and ask when the
  requirement is materially unclear.
- Prefer the simplest implementation that solves the stated problem. Do not add
  speculative features, single-use abstractions, unrequested configurability, or
  broad error handling for scenarios that cannot occur in the current design.
- Make surgical changes. Touch only lines that directly serve the request,
  match the existing style, and do not refactor adjacent code unless asked.
- Clean up only your own changes. Remove imports, variables, helpers, or files
  made unused by the current edit. Mention unrelated dead code instead of
  deleting it.
- Preserve user changes in a dirty worktree. Never revert changes you did not
  make unless explicitly asked.
- Define success criteria for non-trivial tasks. Use a short plan with a
  verification step for each change, then run targeted checks or state why they
  could not be run.
- Push only when explicitly asked.

## Canonical Layout

```text
src/
  gdfm/                  canonical GDFM model core and backbones
  mp_config/             ConfigDict defaults and override helpers
  mp_data/               canonical data loading and subset logic
  mp_metrics/            metric computation from tensors or features
  mp_artifacts/          run layout, config snapshots, checkpoints, tensors
  mp_training/           shared training loop scaffolding
  mp_runners/            CLI workflows and saved-run checkpoint evaluation
  mp_visualization.py    figure generation from saved tensors and records

prototyping/             prototype fallback and transitional reference code
archives/                zipped snapshots of removed experiments/reference trees
baselines/               vendored comparison wrappers
third_party/             preserved external code
```

## Canonical Ownership

- `src/gdfm/`
  owns model-intrinsic code: GDFM math, backbone setup, backbones, and future
  sampling or conversion helpers.
- `src/mp_*`
  owns reusable non-model study infrastructure: configs, data loading,
  artifacts, metrics, runners, visualization, and training scaffolding.
- `src/mp_metrics/`
  computes metric values from in-memory tensors or feature matrices only.
  Checkpoint replay belongs in `mp_runners`; figure writing belongs in
  `mp_visualization`; artifact layout belongs in `mp_artifacts`; tracker
  payload formatting belongs in `mp_tracking`.
- `prototyping/`
  owns fallback and transition code. Keep it readable and stable while the
  canonical path hardens.
- `archives/`
  owns preserved historical snapshots. Treat them as context, not live import
  surfaces.
- `baselines/` and `third_party/`
  are external or vendored code. Treat them as read-only unless explicitly
  asked.

## GDFM Safety Rules

- Keep `prototyping/gdfm_components.py` intact as the reference fallback while
  canonical GDFM evolves in `src/gdfm/core.py`.
- Keep prototype-vs-canonical parity tests green whenever canonical GDFM
  changes.
- Do not bake `prediction_space` or `loss_space` semantics into backbone
  networks. The network stays architecture-only.
- Use GDFM notation explicitly: `eps`, `z_t`, `v_t`, `prediction_space`,
  `loss_space`, `t = 0` noise, and `t = 1` data.

## Data and Artifact Rules

- Persist the exact training tensor or subset artifact for every run.
- Persist the exact `z_init` used for checkpoint replay.
- Post-hoc evaluation must operate on saved artifacts, not rebuilt datasets.
- Keep outputs inside the caller's chosen run directory.
- Use `torch.utils.data.Dataset` and batched loading for real training paths.
- Avoid training-time augmentation in canonical memorization studies unless the
  study is explicitly about augmentation.
- When W&B is enabled and no explicit W&B flag is passed, derive project, run
  name, and group from the resolved config. Project is dataset plus model. Run
  name includes dataset, model, `N`, batch cap, prediction/loss spaces,
  training steps, and seed. Group is dataset plus model plus training steps so
  `x`, `eps`, `v`, batch-size, seed, and `N` sweeps stay comparable.

## Python Style

- Canonical first-party Python follows `PYTHON_STYLE.md`.
- The cleaned `prototyping/gdfm_components.py` file is the style and notation
  exemplar.
- Prefer math-first, teaching-first comments and docstrings.
- Use paper notation directly when the paper defines the quantity: `rho`, `r`,
  `d1`, `d2`, `N`, `tau`, and `t`.
- Use canonical metric keys in return dictionaries, JSON, and W&B logs. Do not
  add duplicate aliases such as `nn_ratio`, `ratios`, or `ratio_threshold` in
  memorization metric records.
- Inline comments for named quantities should state the formula or definition,
  for example `r = d1 / d2` or `delta_f_mem = f_mem(gen) - f_mem(held-out)`.
- Canonical first-party Python should not use type hints or
  `from __future__ import annotations`.

## Environment

- If a checkout has not been installed in editable mode, use `PYTHONPATH=src`
  for direct local runs.
- Before running project-dependent Python commands on the cluster, load and
  activate the project environment with `module load Mambaforge` and
  `mamba activate mem_param_env`.
- The bare system Python is not the project environment.
- Berzelius/NSC nodes may require staging datasets to local scratch.
- Pre-download weights before job submission; compute nodes may have no
  internet.

## Do Not

- Do not recreate a live dependency from `src/` into archived material.
- Do not create vague catch-all modules such as `utils.py` unless there is a
  very clear, narrow purpose.
- Do not edit vendored code under `baselines/*/vendor` or `third_party/`
  unless explicitly asked.
- Do not mutate unrelated files, generated artifacts, notebooks, job scripts, or
  experiment outputs unless explicitly requested.
- Do not push to remote or amend past commits without explicit instruction.

## Workflow

1. Use `prototyping/` or `archives/` for context when needed.
2. Promote stable model or infrastructure code into `src/`.
3. Keep the prototype tests, sampler tests, model-setup tests, canonical GDFM
   tests, and parity tests green before trusting a migration step.
4. Use targeted verification such as `git diff --check`, focused tests, or
   `python -m py_compile` for touched Python files.
5. After meaningful work, append or update `CODING_JOURNAL.md` with a UTC ISO
   timestamp, decisions, commands/checks, results, blockers, and next action,
   unless the task explicitly says not to update the journal.
