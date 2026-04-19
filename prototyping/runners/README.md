# Prototype Runner Lab

This folder is a release-clean lab for paper-relevant runner components. The
synthetic MLP runner keeps a direct, visible training loop. Image runners keep a
small recipe shape and delegate full training/evaluation to the canonical
`src/` stack.

Each runner is meant to be read top to bottom: knobs, config, data, model,
training, metrics, and artifacts.

## Runner Matrix

| Name | Dataset | Backbone | Style | Module | Notebook |
| --- | --- | --- | --- | --- | --- |
| `synthetic_mlp` | `gmm_subspace` | `mlp` | direct script | `prototyping.runners.synthetic_mlp` | `notebooks/synthetic_mlp.ipynb` |
| `image32_unet` | `mnist`, `celeba32_grey`, `flowers102`, `flowers102_128`, `flowers102_256`, `afhq_v2`, `ffhq128` | `unet` | recipe functions | `prototyping.runners.image32_unet` | `notebooks/image32_unet.ipynb` |

## Command Examples

Use the project environment before launching runner checks:

```bash
module load Mambaforge
mamba activate mem_param_env
```

Synthetic direct run:

```bash
PYTHONPATH=src:. python -m prototyping.runners.synthetic_mlp \
  --device cpu \
  --out-dir outputs/prototype_runners/synthetic_mlp \
  --prediction-space v \
  --loss-space v \
  --batch-cap 64 \
  --n-steps 200 \
  --metric-every 50 \
  --ckpt-every 50
```

Image32 recipe run:

```bash
PYTHONPATH=src:. python -m prototyping.runners.image32_unet \
  --device cpu \
  --out-dir outputs/prototype_runners/image32_unet \
  --dataset celeba32_grey \
  --prediction-space v \
  --loss-space v \
  --batch-cap 64 \
  --n-steps 200 \
  --ckpt-every 50 \
  --no-download

# Run a lightweight sanity check before deciding on a full job.
PYTHONPATH=src:. python -m prototyping.runners.image32_unet \
  --dataset flowers102_128 \
  --sanity-updates 2 \
  --sanity-only \
  --batch-cap 4 \
  --n-train 8 \
  --n-eval 8
```

The public runner CLI uses `--prediction-space`, `--loss-space`,
`--batch-cap`, and `--out-dir`. Legacy aliases and validate-only modes have
been removed; component checks now live in tests and notebooks.

## Artifacts

The synthetic runner writes:

- `metrics/train_history.json`
- `metrics/metrics_along_training.json`
- `metrics/final_samples*.pt`
- `figures/*.png`
- `checkpoints/step_*.pt`
- `status.json`

Image runners write the canonical attempt artifact tree under `--out-dir`.
The Image32 runner also writes `metrics/loss_eval.json` with random-`t`
training-style loss probes and fixed-`t` probes at `0.1, 0.3, 0.5, 0.7, 0.9`.

## Notebook Map

The notebooks in `notebooks/` import their sibling runner modules and mirror
the same steps interactively. They cover config construction, data inspection,
model construction, one optimizer update, metric inspection, optional full
runs, and artifact loading.
