# Runner Notebooks And Examples

The active runners now live in `src/mp_runners`. This folder keeps the runner
notebooks and command examples only.

Each runner module is meant to be read top to bottom: knobs, config, data,
model, training, metrics, and artifacts.

## Runner Matrix

| Name | Dataset | Backbone | Style | Module | Notebook |
| --- | --- | --- | --- | --- | --- |
| `synthetic_train_runner` | `gmm_subspace`, `torus3d` | `mlp` | recipe functions | `mp_runners.synthetic_train_runner` | `notebooks/synthetic_train_runner.ipynb` |
| `image_train_runner` | `mnist`, `celeba`, `flowers`, `afhq`, `ffhq` | `unet` | recipe functions | `mp_runners.image_train_runner` | `notebooks/image_train_runner.ipynb` |

## Command Examples

Use the project environment before launching runner checks:

```bash
module load Mambaforge
mamba activate mem_param_env
```

Synthetic recipe run:

```bash
PYTHONPATH=src:. python -m mp_runners.synthetic_train_runner \
  --device cpu \
  --out_dir outputs/runners/synthetic_train_runner \
  --prediction_space v \
  --loss_space v \
  --ambient_dim 8 \
  --intrinsic_data_dim 2 \
  --batch_cap 64 \
  --n_training_steps 200 \
  --ckpt_every 50

# Run a tiny train smoke.
PYTHONPATH=src:. python -m mp_runners.synthetic_train_runner \
  --device cpu \
  --dataset torus3d \
  --ambient_dim 3 \
  --n_train 16 \
  --n_test 32 \
  --batch_cap 8 \
  --n_training_steps 2 \
  --ckpt_every 1

# Replay synthetic checkpoint metrics later from a finished training attempt.
PYTHONPATH=src:. python -m mp_runners.eval_synthetic \
  --device cuda \
  --run_dir outputs/runners/synthetic_train_runner/runs/.../DONE_...
```

Image recipe run:

```bash
PYTHONPATH=src:. python -m mp_runners.image_train_runner \
  --device cuda \
  --out_dir outputs/runners/image_train_runner \
  --dataset celeba \
  --image_size 32 \
  --model unet \
  --prediction_space v \
  --loss_space v \
  --batch_cap 64 \
  --n_training_steps 200 \
  --ckpt_every 50 \
  --no_download

# Run the same train pipeline with tiny step counts before submitting a long job.
PYTHONPATH=src:. python -m mp_runners.image_train_runner \
  --device cuda \
  --dataset flowers \
  --image_size 128 \
  --model unet \
  --generation_monitor \
  --generation_monitor_samples 8 \
  --batch_cap 4 \
  --n_train 8 \
  --n_test 8 \
  --n_training_steps 2 \
  --ckpt_every 1
```

The image runner CLI is intentionally slim. Public dataset names are only
`mnist`, `celeba`, `flowers`, `afhq`, and `ffhq`; `--image_size` is the only
public resolution control. For example, `--dataset flowers --image_size 64
--model unet` resolves internally to the matching Flowers102 loader/profile.

`image_train_runner` is the single-process image entrypoint. Future multi-GPU/DDP
experiments should live in a separate script such as `image_train_runner_ddp`, launched
with `torchrun`, so distributed concerns do not leak into this runner.

The core image-runner size knobs mean exactly this:

| Flag | Config field | Meaning |
| --- | --- | --- |
| `--dataset` | `config.study.dataset_key`, `config.data.dataset` | Public dataset choice plus internal loader name. |
| `--image_size` | `config.data.image_size` | Image resolution; standard CLI sizes are 32, 64, 128, and 256. |
| `--model` | `config.network.arch` | Model family. Currently `unet`. |
| `--n_train` | `config.data.n_train` | Number of real images used for optimization. |
| `--n_test` | `config.data.n_test` | Number of real held-out/reference images materialized for later tests. |
| `--generation_monitor_samples` | `config.generation_monitor.n_samples` | Number of fixed-noise samples available to train-time image monitors. |
| `--monitor_batch_size` | `config.monitoring.batch_size` | No-grad batch size for train-time image monitors; defaults to `--batch_cap`. |
| `--sampling_steps` | `config.sampling.n_steps` | Number of solver steps used to generate images from `z_init`. |
| `--precision` | `config.runtime.precision` | Numerical mode: `fp32`, CUDA `tf32`, or CUDA `bf16` autocast. |

W&B projects default to the user-facing dataset key plus image size plus model
arch: `memparam-flowers-128-unet`, `memparam-mnist-32-unet`, etc. The old broad
`memparam-image32-unet` value is treated as auto so copied commands stop mixing
datasets into one W&B project. Default W&B run names include dataset, model,
`N`, batch cap, prediction/loss spaces, training steps, and seed. Default W&B
groups include only dataset, model, and training steps, so `x`, `eps`, `v`,
batch-size, seed, and `N` sweeps stay comparable. Explicit
`--wandb_project`, `--wandb_name`, and `--wandb_group` flags override these
automatic values.

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

Image runners write the canonical attempt artifact tree under `--out_dir`.
Training always saves the config snapshot, train/reference tensors, `z_init.pt`,
train history, checkpoints, and generation-monitor figures when enabled.
Checkpoint evaluation is run separately after training and writes
`metrics/loss_eval.json` with held-out random-`t`
loss probes and fixed-`t` probes at `0.5` and `0.75`, plus
`metrics/checkpoint_sample_previews.json` and preview tensors under
`metrics/checkpoint_sample_previews/`. When `--generation_monitor` is enabled,
image runners save `artifacts/z_init.pt` once and sample from that same fixed
noise bank at step 0, every checkpoint, and the final model. The panels are
written under `figures/generation_monitor_*.png` and logged to W&B as
`generation_monitor/samples` for checkpoint panels and
`generation_monitor/final_samples` for the final model when tracking is active.
Canonical attempt folders receive a `DONE_` prefix after training artifacts are
successfully written, so completed-training folders are visible from the
filesystem while failed or aborted attempts keep their original names.

Synthetic MLP runners now use the same checkpoint-first lifecycle. Training
always saves the canonical attempt tree plus controlled geometry. Checkpoint
metrics are replayed later with `python -m mp_runners.eval_synthetic --run_dir
<DONE_...>`. Public synthetic metric names use `intrinsic_*` for true
low-dimensional metric coordinates across both projected GMM and torus data.
For projected GMM runs, `--ambient_dim` is the observed vector dimension `D`
and `--intrinsic_data_dim` is the controlled linear subspace dimension `d`;
legacy `data_dim`/`intrinsic_dim` public aliases were removed. For
`torus3d`, `--ambient_dim` embeds a canonical 3D torus into observed dimension
`D`; the local manifold dimension is fixed at 2 and `--intrinsic_data_dim` is
rejected because there is no configurable global linear intrinsic space.
Held-out true-data checks use the `holdout_*` prefix. Old `latent_*`,
`baseline_*`, and component-coverage metric aliases are removed.

## Experiment Family And Sampler Scope

The active paper runners should be read as FM experiments with deterministic
ODE sampling. Both train runners set `config.framework.family = "fm"`: image
runs do this in the image config builder, and synthetic runs do this in the
synthetic train runner. Image generation uses deterministic Heun sampling from
the saved fixed `z_init` bank. Synthetic generation exposes only deterministic
Euler or Heun sampling through `--sampling_method`.

Lower-level VP helpers still exist in `gdfm.samplers` for tests and examples,
including `vp_sde` / VP reverse-SDE sampling. They are not part of normal
image or synthetic train/eval runs, and current runner outputs should not be
described as DDPM results. If a future study needs VP or DDPM behavior, it
should get an explicit runner/API surface rather than being implied by the
current FM runs.

## Bonnaire-Close Image32 Defaults

The default 32px/Image32 profile is kept close to Bonnaire's UNet setup:
`base_channels=32`, `channel_mult=(1, 2, 4)`, `num_res_blocks=2`,
`dropout=0.1`, `time_emb_dim=None` (resolved by the model to
`base_channels * 4`), and `attn_resolutions=(16, 8)`. This is a
Bonnaire-close model/data recipe, not a DDPM reproduction. Training keeps Adam
with `lr=1e-4`, the framework family remains continuous-flow matching, and
generation uses deterministic ODE sampling.

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
