# FFHQ64 Parameterization Sweep

Purpose: short preflight runs before any 1M-step FFHQ experiments. This checks
FFHQ64 config resolution, output layout, W&B run naming, `param` filtering, and
train-time monitor logging across diagonal parameterizations.

Fixed image setting:

- dataset: `ffhq`
- image size: `64`
- model: `unet`
- `n_train=4000`
- `n_test=1024`
- seed: `0`

Grid:

- diagonal parameterizations: `x/x`, `eps/eps`, `v/v`

This gives 3 runs. All runs use:

- output root: `outputs/job_sets/ffhq64_param_sweep_20260426/`
- W&B project: `memparam-ffhq64_param_sweep_20260426`
- W&B group: `ffhq64_param_sweep_20260426`
- Slurm logs: `jobs/job_sets/ffhq64_param_sweep_20260426/slurm/`

Submit the array:

```bash
sbatch jobs/job_sets/ffhq64_param_sweep_20260426/submit.sh
```

Run one short interactive debug job:

```bash
bash jobs/job_sets/ffhq64_param_sweep_20260426/debug_one.sh v 0
```

The main script defaults to `N_TRAINING_STEPS=2000` and `CKPT_EVERY=500`.
Override those at submission time if needed, for example:

```bash
N_TRAINING_STEPS=5000 CKPT_EVERY=1000 sbatch jobs/job_sets/ffhq64_param_sweep_20260426/submit.sh
```
