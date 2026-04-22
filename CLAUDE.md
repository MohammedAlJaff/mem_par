# mem_par Repo Guide

Paper: "Equivalent in Principle, Unequal in Practice"

Current repo question:
- under fixed architecture, data, and training protocol, how do target
  parameterizations (`x`, `eps`, `v`) change memorization dynamics?

Communication rule:
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

Key measurements:
- `f_mem`
- nearest-neighbor ratio and distance statistics
- checkpoint-level memorization trajectories over training

---

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

---

## Canonical Ownership

- `src/gdfm/`
  Owns model-specific code only: canonical GDFM, backbone setup, backbones, and
  future sampling/conversion helpers.

- `src/mp_*`
  Owns reusable but non-model-specific support code: configs, data, artifacts,
  metric computation, runner evaluation, visualization, and training skeletons.

- `src/mp_metrics/`
  Computes metric values from in-memory tensors or feature matrices only.
  Checkpoint replay belongs in `mp_runners`; figure writing belongs in
  `mp_visualization`; artifact layout belongs in `mp_artifacts`; tracker
  payload formatting belongs in `mp_tracking`.

- `prototyping/`
  Owns fallback and transition code. Keep it readable and stable while the
  canonical path hardens.

- `archives/`
  Owns preserved zip snapshots of the removed `experiments/` and `reference/`
  trees. Treat them as history, not as live import surfaces.

---

## Active Entry Points

There is intentionally no live first-party `experiments/` tree right now.

The active surface is the reusable code in `src/`, plus the fallback prototype
surface in `prototyping/`.

If a checkout has not been installed in editable mode yet, use `PYTHONPATH=src`
for direct local runs.

---

## GDFM Status

- Prototype fallback:
  `prototyping/gdfm_components.py`

- Canonical destination:
  `src/gdfm/core.py`

- Canonical backbone factory:
  `src/gdfm/model_setup.py`

- Canonical backbones:
  `src/gdfm/networks/`

Transition rule:
- the prototype remains the scientific safety reference
- canonical GDFM evolves in `src/gdfm/`
- parity tests protect the move

The backbone network should stay architecture-only. Parameterization semantics
belong to GDFM, not to the MLP, U-Net, or JiT modules.

---

## Data And Artifact Rules

- Persist the exact training tensor or subset artifact for every run.
- Persist the exact `z_init` used for checkpoint replay.
- Post-hoc evaluation must operate on saved artifacts, not on rebuilt datasets.
- Keep outputs inside the caller's chosen run directory.
- Use `torch.utils.data.Dataset` and batched loading for real training paths.
- Avoid training-time augmentation in canonical memorization studies unless the
  study is explicitly about augmentation.
- When W&B is enabled and no explicit W&B flag is passed, derive project, run
  name, and group from the resolved config. Project is dataset+model. Run name
  includes dataset, model, `N`, batch cap, prediction/loss spaces, training
  steps, and seed. Group is dataset+model+training steps so `x`, `eps`, `v`,
  batch-size, seed, and `N` sweeps stay comparable.

---

## Python Style

- Canonical first-party Python follows `PYTHON_STYLE.md`.
- The cleaned `prototyping/gdfm_components.py` file is the style and notation
  exemplar.
- Prefer math-first, teaching-first comments and docstrings.
- Use paper notation directly in code when the paper defines the quantity:
  `rho`, `r`, `d1`, `d2`, `N`, `tau`, and `t`.
- Use canonical metric keys in return dictionaries, JSON, and W&B logs. Do not
  add duplicate aliases such as `nn_ratio`, `ratios`, or `ratio_threshold` in
  memorization metric records.
- Inline comments for named quantities should state the formula or definition,
  for example `r = d1 / d2` or `delta_f_mem = f_mem(gen) - f_mem(held-out)`.
- No type hints in canonical first-party Python.
- Use GDFM notation explicitly: `eps`, `z_t`, `v_t`, `prediction_space`,
  `loss_space`, `t = 0` noise, `t = 1` data.

---

## Non-Canonical Areas

- `archives/` contains preserved context, not live code.
- `baselines/*/vendor` and `third_party/` are read-only unless explicitly
  instructed otherwise.

---

## Workflow

1. Use `prototyping/` or `archives/` for context when needed.
2. Promote stable reusable code into `src/gdfm/` or the relevant `src/mp_*`
   package.
3. Keep the prototype tests, sampler tests, model-setup tests, canonical GDFM
   tests, and parity tests green before trusting a migration step.
4. Push only when explicitly asked.

---

## Cluster Notes

- Berzelius/NSC nodes may require staging datasets to local scratch.
- Pre-download weights before job submission; compute nodes may have no
  internet.
- Prefer study-local manifests, figures, and outputs over repo-global output
  conventions.
