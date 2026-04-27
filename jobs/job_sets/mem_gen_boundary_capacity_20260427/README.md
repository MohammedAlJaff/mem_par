# Mem-Gen-Boundary Capacity Control

Purpose: focused CelebA32 grayscale capacity-control sweep for checking whether
the memorization/generalization boundary from the main `mem-gen-boundary` sweep
is stable under larger UNet capacity.

Fixed setting:

- dataset: `celeba`
- image size: `32`
- recipe: `bonnaire_celeba32`
- optimizer: `adam`
- `batch_cap=256`
- `n_test=1024`
- checkpoint schedule: `bonnaire-1m`
- logical target: `1_000_000` optimizer steps

Grid:

- `n_train`: `256`, `2048`, `16384`
- seeds: `0`, `1`, `2`
- parameterizations: `x/x`, `eps/eps`, `v/v`, `x/v`
- model capacities:
  - `base32_m124`: `base_channels=32`, `channel_mult=1,2,4`, `3,917,409` params
  - `base64_m124`: `base_channels=64`, `channel_mult=1,2,4`, `15,637,697` params
  - `base64_m1244`: `base_channels=64`, `channel_mult=1,2,4,4`, `25,810,369` params

This gives `3 * 3 * 4 * 3 = 108` runs.

Run layout:

- output root: `outputs/job_sets/mem_gen_boundary_capacity_20260427/`
- W&B project: `memparam-mem_gen_boundary_capacity_20260427`
- W&B group: `mem_gen_boundary_capacity_20260427`
- Slurm logs: `jobs/job_sets/mem_gen_boundary_capacity_20260427/slurm/`

Submit the full array:

```bash
sbatch jobs/job_sets/mem_gen_boundary_capacity_20260427/submit.sh
```

Run one short offline smoke task:

```bash
WANDB_MODE=offline N_TRAINING_STEPS=10 CKPT_SCHEDULE=interval CKPT_EVERY=5 \
  sbatch --array=0 jobs/job_sets/mem_gen_boundary_capacity_20260427/submit.sh
```

Resume one timed-out or killed task from its latest attempt:

```bash
RESUME_FROM_ATTEMPT=/path/to/attempt \
  sbatch --array=<task_id> jobs/job_sets/mem_gen_boundary_capacity_20260427/submit.sh
```

Resume keeps `n_training_steps=1000000` as the logical target, uses absolute
checkpoint step numbers, and continues the source W&B run when the source
attempt has a saved W&B run id.
