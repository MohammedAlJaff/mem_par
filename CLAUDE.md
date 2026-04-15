# mem_par Repo Guide

Paper: "Equivalent in Principle, Unequal in Practice"

Current repo question:
- under fixed architecture, data, and training protocol, how do target
  parameterizations (`x`, `eps`, `v`) change memorization dynamics?

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
  mp_metrics/            memorization metrics and checkpoint replay
  mp_artifacts/          run layout, config snapshots, checkpoints, tensors
  mp_training/           shared training loop scaffolding

experiments/
  mnist_n4_diagonal/     canonical MNIST proof-of-concept study
  mnist_nsweep/          canonical MNIST training-set-size sweep
  gmm_subspace/          canonical high-dimensional synthetic study
  celeba32_grey/         canonical real-image study
  legacy/                preserved pre-reset experiment stack

reference/
  walkthroughs/          pedagogical runnable scripts
  sanity/                smoke checks and teaching notebooks
  notebooks/             exploratory notebooks
  archive/               older preserved reference code

prototyping/             prototype fallback and transitional reference code
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
  memorization metrics, and training skeletons.

- `experiments/<study>/`
  Owns study-specific choices, entrypoints, and outputs.

- `reference/`
  Owns pedagogy, sanity checks, and exploratory context. Useful, but
  non-canonical.

- `prototyping/`
  Owns fallback and transition code. Keep it readable and stable while the
  canonical path hardens.

---

## Active Entry Points

Canonical study commands:

```bash
python -m experiments.mnist_n4_diagonal.train --help
python -m experiments.mnist_n4_diagonal.evaluate --help
python -m experiments.mnist_n4_diagonal.plot --help
python -m experiments.mnist_nsweep.train --help
python -m experiments.gmm_subspace.train --help
python -m experiments.celeba32_grey.train --help
```

Reference material:

```bash
python reference/walkthroughs/01_primitives.py
python reference/sanity/check_3x3_grid.py
python reference/sanity/sanity_nn_stats.py
```

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

## Experiment Contract

Each active experiment folder should stay thin and self-contained:

- `config.py`
- `train.py`
- `evaluate.py`
- `plot.py`
- `runs/`
- `figures/`

Use `src/` packages for reusable logic. Do not grow a second parallel
infrastructure inside the experiment folder.

---

## Data And Artifact Rules

- Persist the exact training tensor or subset artifact for every run.
- Persist the exact `z_init` used for checkpoint replay.
- Post-hoc evaluation must operate on saved artifacts, not on rebuilt datasets.
- Keep outputs inside the owning experiment folder.
- Use `torch.utils.data.Dataset` and batched loading for real training paths.
- Avoid training-time augmentation in canonical memorization studies unless the
  study is explicitly about augmentation.

---

## Python Style

- Canonical first-party Python follows `PYTHON_STYLE.md`.
- The cleaned `prototyping/gdfm_components.py` file is the style and notation
  exemplar.
- Prefer math-first, teaching-first comments and docstrings.
- No type hints in canonical first-party Python.
- Use GDFM notation explicitly: `eps`, `z_t`, `v_t`, `prediction_space`,
  `loss_space`, `t = 0` noise, `t = 1` data.

---

## Non-Canonical Areas

- `reference/archive/` is preserved context, not the source of truth.
- `experiments/legacy/` is preserved legacy infrastructure, not the source of
  truth.
- `experiments.synthetic` and `experiments.natural_images` are compatibility
  shims for the archived stack.
- `baselines/*/vendor` and `third_party/` are read-only unless explicitly
  instructed otherwise.

---

## Workflow

1. Explore or explain ideas in `reference/` or `prototyping/`.
2. Promote stable reusable code into `src/gdfm/` or the relevant `src/mp_*`
   package.
3. Expose paper-facing study logic in `experiments/<study>/`.
4. Keep the prototype tests, sampler tests, model-setup tests, canonical GDFM
   tests, and parity tests green before trusting a migration step.
5. Push only when explicitly asked.

---

## Cluster Notes

- Berzelius/NSC nodes may require staging datasets to local scratch.
- Pre-download weights before job submission; compute nodes may have no
  internet.
- Prefer study-local manifests, figures, and outputs over repo-global output
  conventions.
