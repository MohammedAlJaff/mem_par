# Mem-Gen-Boundary CelebA32 Sweep

Purpose: core CelebA32 grayscale memorization/generalization boundary runs,
using the Bonnaire-compatible recipe and long 1M-step training horizon.

Fixed image setting:

- dataset: `celeba`
- image size: `32`
- recipe: `bonnaire_celeba32`
- model: `unet`
- `n_test=1024`

Grid:

- `N_train`: `256`, `512`, `1024`, `2048`, `4096`, `8192`, `16384`
- seeds: `0`, `1`, `2`
- parameterizations: `x/x`, `eps/eps`, `v/v`, `x/v`

This gives 84 runs. All runs use:

- output root: `outputs/job_sets/mem_gen_boundary_20260427/`
- W&B project: `memparam-mem_gen_boundary_20260427`
- W&B group: `mem_gen_boundary_20260427`
- Slurm logs: `jobs/job_sets/mem_gen_boundary_20260427/slurm/`
- batch cap: `256`
- checkpoint schedule: `bonnaire-1m`
- logical training target: `1_000_000` optimizer steps

Submit the full array:

```bash
sbatch jobs/job_sets/mem_gen_boundary_20260427/submit.sh
```

Run one short offline smoke task:

```bash
WANDB_MODE=offline N_TRAINING_STEPS=10 CKPT_SCHEDULE=interval CKPT_EVERY=5 \
  sbatch --array=0 jobs/job_sets/mem_gen_boundary_20260427/submit.sh
```

Resume one timed-out or killed task from its latest attempt:

```bash
RESUME_FROM_ATTEMPT=/path/to/attempt \
  sbatch --array=<task_id> jobs/job_sets/mem_gen_boundary_20260427/submit.sh
```

Resume keeps `n_training_steps=1000000` as the logical target, uses absolute
checkpoint step numbers, and continues the source W&B run when the source
attempt has a saved W&B run id.
