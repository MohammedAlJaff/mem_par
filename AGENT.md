# AGENT Guide

Short operational guide for agents and contributors working in this repo.

## North Star

- Optimize for experiment ownership over framework cleverness.
- Duplication between experiments is acceptable.
- Do not rebuild a repo-wide abstraction layer.

## Canonical Placement

- `experiments/synthetic/low_d/`
  Use for synthetic systems whose geometry is directly inspectable in 2D or 3D.

- `experiments/synthetic/high_d/`
  Use for synthetic systems with known latent structure in high ambient dimension.

- `experiments/synthetic/mnist/`
  Use for image-like synthetic data where the latent generator is known and controlled.

- `experiments/natural_images/`
  Use for real observational image datasets such as CelebA or AFHQ.

## Routing Rule

- If the generator is known and controlled, it is `synthetic`.
- If the data comes from the world, it is `natural_images`.
- If it is synthetic and visually geometric, it is `low_d`.
- If it is synthetic and structurally high-dimensional, it is `high_d`.
- If it is synthetic and image-like with known latent generation, it is `mnist`.

## Required Experiment Shape

Every canonical experiment owns:

- `README.md`
- `config.yaml`
- `paths.example.yaml`
- `run.py`
- `sweep.py`
- `aggregate.py`
- `runs/`
- `figures/`

Additional expectations:

- `synthetic/high_d/` and `synthetic/mnist/`
  Prefer `data_model.py`, `dataset.py`, and `ground_truth.py` once the experiment is more than a tiny demo.

- `synthetic/mnist/`
  Add `renderer.py` when image formation is part of the scientific claim.

- `natural_images/`
  Keep experiment-specific dataset assumptions inside the experiment folder, even when the family uses shared loaders or models.

## Allowed Shared Code

Shared code may live only in:

- `experiments/synthetic/low_d/common/`
- `experiments/synthetic/high_d/common/`
- `experiments/synthetic/mnist/common/`
- `experiments/natural_images/common/`

## Do Not

- Do not create `experiments/synthetic/common/`.
- Do not revive `src/`, root `configs/`, root `scripts/`, or a repo-wide pipeline.
- Do not use `_base_` inheritance in canonical configs.
- Do not add a global experiment registry or generic dispatch layer.
- Do not put synthetic image experiments in `natural_images/`.
- Do not import canonical experiment code from `lab/reference/`.
- Do not edit vendored code under `baselines/*/vendor` or `third_party/` unless explicitly asked.

## Artifact Rules

- Save the dataset artifact immediately after `build_dataset()`.
- Evaluate against `load_train_data(run_dir)`, never against a rebuilt dataset.
- Keep outputs local to the owning experiment folder.
- Encode the active sweep axes in run directory names instead of assuming one universal pattern.

## Promotion Workflow

1. Prototype in `lab/`.
2. Promote stable work into the correct canonical experiment folder.
3. If code is reused within one subfamily, consider that subfamily's `common/`.
4. If code might be shared across subfamilies, duplicate first unless the abstraction is clearly stable.
