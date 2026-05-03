# Mem-Gen-Boundary CelebA32 SGD+Momentum Sweep

Purpose: optimizer-control sister set for `mem_gen_boundary_20260427`.
This mirrors the Adam CelebA32 memorization/generalization boundary grid while
using Bonnaire-style SGD with momentum.

Fixed image setting:

- dataset: `celeba`
- image size: `32`
- recipe: `bonnaire_celeba32`
- model: `unet`
- `n_test=1024`
- optimizer: `sgd`
- learning rate: `0.01`
- momentum: `0.95`
- weight decay: `0.0`

Grid:

- `N_train`: `256`, `512`, `1024`, `2048`, `4096`, `8192`, `16384`
- seeds: `0`, `1`, `2`
- parameterizations: `x/x`, `eps/eps`, `v/v`, `x/v`

This gives 84 runs. All runs use:

- output root: `outputs/job_sets/mem_gen_boundary_sgdm_20260427/`
- W&B project: `memparam-mem_gen_boundary_sgdm_20260427`
- W&B group: `mem_gen_boundary_sgdm_20260427`
- Slurm logs: `jobs/job_sets/mem_gen_boundary_sgdm_20260427/slurm/`
- batch cap: `256`
- checkpoint schedule: `bonnaire-1m`
- logical training target: `1_000_000` optimizer steps

Capacity sanity check:

- active CelebA32 UNet: `3,917,409` trainable parameters
- vendored Bonnaire `run_Unet.py -W 32`: `4,055,233` total /
  `4,023,233` trainable parameters
- conclusion: same capacity scale, not byte-identical architecture

Required smoke gate before full submission:

```bash
WANDB_MODE=offline N_TRAINING_STEPS=2000 CKPT_SCHEDULE=interval CKPT_EVERY=500 \
  sbatch --array=0 jobs/job_sets/mem_gen_boundary_sgdm_20260427/submit.sh
```

Smoke acceptance:

- task reaches final `2000/2000`
- no `Traceback`, OOM, `nan`, or runtime error signatures
- checkpoints and monitor figures are written
- stdout confirms `optimizer=sgd`, `lr=0.01`, and `momentum=0.95`

Submit the full array only after the smoke passes:

```bash
sbatch jobs/job_sets/mem_gen_boundary_sgdm_20260427/submit.sh
```

Resume one timed-out or killed task from its latest attempt:

```bash
RESUME_FROM_ATTEMPT=/path/to/attempt \
  sbatch --array=<task_id> jobs/job_sets/mem_gen_boundary_sgdm_20260427/submit.sh
```

Resume keeps `n_training_steps=1000000` as the logical target, uses absolute
checkpoint step numbers, and continues the source W&B run when the source
attempt has a saved W&B run id.
