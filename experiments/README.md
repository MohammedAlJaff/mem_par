# Experiments

This tree is the canonical home for study-specific code and scripts.

## Active Studies

- `mnist_n4_diagonal/`
  Proof-of-concept MNIST diagonal comparison with tiny `N`
- `mnist_nsweep/`
  MNIST sweep over training-set size
- `gmm_subspace/`
  High-dimensional synthetic GMM on a linear subspace
- `celeba32_grey/`
  CelebA 32x32 greyscale image study

Each active study stays thin and owns:

- `config.py`
- `train.py`
- `evaluate.py`
- `plot.py`
- local `runs/`
- local `figures/`

## Shared Code Boundary

Shared reusable machinery belongs in `src/`, not in ad hoc experiment-local
helpers:

- `src/gdfm/` for model-specific code
- `src/mp_*` for shared config, data, artifacts, metrics, and training helpers

`experiments/_shared/` is only for small experiment-surface helpers such as CLI
glue or study-facing plotting wrappers.

## Legacy Stack

Older experiment code is preserved under `experiments/legacy/`.

The packages `experiments.synthetic` and `experiments.natural_images` remain as
compatibility shims so archived imports still resolve, but new first-party work
should not be added there.
