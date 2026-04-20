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
| `image_runner` | `mnist`, `celeba32_grey`, `flowers102`, `flowers102_128`, `flowers102_256`, `afhq_v2`, `ffhq128` | `unet` | recipe functions | `prototyping.runners.image_runner` | `notebooks/image_runner.ipynb` |

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

Image recipe run:

```bash
PYTHONPATH=src:. python -m prototyping.runners.image_runner \
  --device cuda \
  --out-dir outputs/prototype_runners/image_runner \
  --dataset celeba32_grey \
  --prediction-space v \
  --loss-space v \
  --batch-cap 64 \
  --n-steps 200 \
  --ckpt-every 50 \
  --no-download

# Run the same full train+eval pipeline with tiny step counts before submitting a long job.
PYTHONPATH=src:. python -m prototyping.runners.image_runner \
  --device cuda \
  --dataset flowers102_128 \
  --generation-trace \
  --generation-trace-count 8 \
  --no-generation-trace-show \
  --batch-cap 4 \
  --n-train 8 \
  --n-eval 8 \
  --n-steps 2 \
  --ckpt-every 1 \
  --final-eval

# Replay checkpoint metrics later from a finished training attempt.
PYTHONPATH=src:. python -m prototyping.runners.image_runner \
  --device cuda \
  --eval-run-dir outputs/prototype_runners/image_runner/runs/.../DONE_...
```

The image runner CLI is intentionally slim. It exposes run identity, dataset/preset,
prediction/loss spaces, selected dataset sizes, training cadence, sampling
count, data root/download, W&B basics, `--generation-trace`, and optional
checkpoint evaluation. Architecture and low-level data variants live in named
presets selected by `--config-preset` or in dataset profiles such as
`flowers102_128`.

`image_runner` is the single-process image entrypoint. Future multi-GPU/DDP
experiments should live in a separate script such as `image_runner_ddp`, launched
with `torchrun`, so distributed concerns do not leak into this runner.

The core image-runner size knobs mean exactly this:

| Flag | Config field | Meaning |
| --- | --- | --- |
| `--n-train` | `config.data.n_train` | Number of real images used for optimization. |
| `--n-eval` | `config.data.n_eval` | Number of real held-out/reference images loaded for evaluation. |
| `--eval-samples` | `config.evaluation.n_samples` | Number of generated samples per checkpoint; also the size of `artifacts/z_init.pt`. |
| `--sampling-steps` | `config.sampling.n_steps` | Number of solver steps used to generate images from `z_init`. |
| `--precision` | `config.runtime.precision` | Numerical mode: `fp32`, CUDA `tf32`, or CUDA `bf16` autocast. |
| `--compare-numerics` | runner-only | Runs an in-memory fp32-vs-selected-precision smoke benchmark instead of a full artifact run. |
| `--final-eval` | runner-only | Runs checkpoint metrics immediately after training. Defaults off for long jobs. |
| `--eval-run-dir` | runner-only | Replays checkpoint metrics later from a finished `DONE_...` attempt directory. |

W&B projects default to the user-facing dataset key plus model arch:
`memparam-flowers102_128-unet`, `memparam-mnist-unet`, etc. The old broad
`memparam-image32-unet` value is treated as auto so copied commands stop mixing
datasets into one W&B project.

Flowers102 profiles intentionally train on the large official `test` split and
use the small official `train` split as the held-out/reference split. The
official `train` split has only 1,020 images; the official `test` split has
6,149.

## Artifacts

The synthetic runner writes:

- `metrics/train_history.json`
- `metrics/metrics_along_training.json`
- `metrics/loss_eval.json`
- `metrics/checkpoint_sample_previews.json`
- `metrics/checkpoint_sample_previews/step_*.pt`
- `metrics/final_samples*.pt`
- `figures/*.png`
- `checkpoints/step_*.pt`
- `status.json`

Image runners write the canonical attempt artifact tree under `--out-dir`.
Training always saves the config snapshot, train/reference tensors, `z_init.pt`,
train history, checkpoints, and generation-trace figures when enabled. Immediate
checkpoint evaluation is opt-in with `--final-eval`; otherwise metrics can be
replayed later with `--eval-run-dir`.
Checkpoint evaluation writes `metrics/loss_eval.json` with held-out random-`t`
loss probes and fixed-`t` probes at `0.5` and `0.75`, plus
`metrics/checkpoint_sample_previews.json` and preview tensors under
`metrics/checkpoint_sample_previews/`. When `--generation-trace` is enabled,
image runners save `artifacts/z_init.pt` once and sample from that same fixed
noise bank at step 0, every checkpoint, and the final model. The panels are
written under `figures/generation_trace_*.png` and logged to W&B as
`generation_trace/samples_over_train` against `generation_trace/step` when
tracking is active.
Canonical attempt folders receive a `DONE_` prefix after training artifacts are
successfully written, so completed-training folders are visible from the
filesystem while failed or aborted attempts keep their original names.

## Bonnaire-Close Image32 Defaults

The default 32px/Image32 profile is kept close to Bonnaire's UNet setup:
`base_channels=32`, `channel_mult=(1, 2, 4)`, `num_res_blocks=2`,
`dropout=0.1`, `time_emb_dim=None` (resolved by the model to
`base_channels * 4`), and `attn_resolutions=(16, 8)`. Training keeps Adam with
`lr=1e-4` and the framework family remains continuous-flow matching.

Bonnaire does use time embeddings: their vendor UNet defines
`SinusoidalPositionEmbeddings` in
`baselines/bonnaire/vendor/Experiments/src/Utils/Unet.py`, expands the time
embedding to `base_channels * 4`, and injects it into each ResNet block.
Remaining intentional differences are that Bonnaire uses integer DDPM
timesteps with epsilon-noise MSE, while this runner uses continuous FM time and
configurable prediction/loss spaces; Bonnaire adds time embeddings into blocks,
while our `SmallUNet`/Bonnaire adapter uses scale/shift conditioning; and the
training loop, artifacts, and W&B tracking stay on the canonical `src/` stack.

## Notebook Map

The notebooks in `notebooks/` import their sibling runner modules and mirror
the same steps interactively. They cover config construction, data inspection,
model construction, metric inspection, optional short/full runs, and artifact
loading.
