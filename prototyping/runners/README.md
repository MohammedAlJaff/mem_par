# Prototype Runner Lab

This folder is a transitional lab for paper-relevant runner components. It is
for fast inspection and iteration. Image runners still delegate full jobs to
the canonical `src/` training/evaluation path, while the synthetic MLP runner
is a direct old-style prototype with its own visible loop and synthetic metrics.

Each runner is meant to be read top to bottom. The synthetic runner is the most
literal version: its file shows output folders, config, manual/canonical GMM
data, model setup, training, periodic metrics, W&B logging, and figures in one
place. Image runners keep the smaller recipe-function shape:

1. knobs
2. config construction
3. data bundle loading
4. model construction
5. one optimizer update
6. metric inspection
7. canonical training and checkpoint evaluation

## Runner Matrix

| Name | Dataset | Backbone | Style | Module | Notebook |
| --- | --- | --- | --- | --- | --- |
| `synthetic_mlp` | synthetic family, default `gmm_subspace` | `mlp` | direct script | `prototyping.runners.synthetic_mlp` | `notebooks/synthetic_mlp.ipynb` |
| `mnist_unet` | `mnist` | `unet` | recipe functions | `prototyping.runners.mnist_unet` | `notebooks/mnist_unet.ipynb` |
| `celeba32_unet` | `celeba32_grey` | `unet` | recipe functions | `prototyping.runners.celeba32_unet` | `notebooks/celeba32_unet.ipynb` |
| `celeba32_jit` | `celeba32_grey` | `jit` | recipe functions | `prototyping.runners.celeba32_jit` | `notebooks/celeba32_jit.ipynb` |

This table is the human map. There is no runtime registry for this lab.

## Command Examples

```bash
PYTHONPATH=src python -m prototyping.runners.synthetic_mlp --validate-only
PYTHONPATH=src python -m prototyping.runners.mnist_unet --validate-only --no-download
PYTHONPATH=src python -m prototyping.runners.celeba32_unet --validate-only --no-download
PYTHONPATH=src python -m prototyping.runners.celeba32_jit --validate-only --no-download
```

To launch a short training/evaluation run, remove `--validate-only` and set
the usual run knobs:

```bash
PYTHONPATH=src python -m prototyping.runners.synthetic_mlp \
  --device cpu \
  --prediction-space v \
  --loss-space v \
  --batch-cap 64 \
  --n-steps 200 \
  --metric-every 50 \
  --ckpt-every 50
```

The synthetic runner writes old-style local artifacts:

- `metrics/train_history.json`
- `metrics/metrics_along_training.json`
- `metrics/final_samples*.pt`
- `figures/*.png`
- `checkpoints/step_*.pt`
- `status.json`

## Notebook Map

The notebooks in `notebooks/` import their sibling runner modules and mirror
the same steps interactively. For synthetic, the notebook mirrors the direct
script flow. For image runners, the notebooks mirror the recipe functions:

- config construction
- data bundle inspection
- model shape and parameter inspection
- one loss/update check
- memorization metric inspection
- command construction
- optional short validation run
- artifact and figure loading

The synthetic MLP notebook currently defaults to projected GMM and also
includes latent projection, residual energy, oracle GMM metrics, component
coverage, and ambient-vs-latent memorization checks.
