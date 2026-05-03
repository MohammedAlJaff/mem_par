# Flowers64 N Sweep

Fixed image setting:

- dataset: `flowers`
- image size: `64`
- model: `unet`
- reference count: `n_test=1020`

Grid:

- `N_train`: `128`, `256`, `512`, `1024`, `2048`, `4096`
- diagonal parameterizations only: `x/x`, `eps/eps`, `v/v`

This gives 18 runs. All runs use:

- output root: `outputs/job_sets/flowers64_n_sweep_20260425/`
- W&B project: `memparam-flowers64_n_sweep_20260425`
- W&B group: `flowers64_n_sweep_20260425`
- Slurm logs: `jobs/job_sets/flowers64_n_sweep_20260425/slurm/`

Submit the full array:

```bash
sbatch jobs/job_sets/flowers64_n_sweep_20260425/submit.sh
```

Run one short interactive debug job:

```bash
bash jobs/job_sets/flowers64_n_sweep_20260425/debug_one.sh 512 v 0
```

The main script defaults to `N_TRAINING_STEPS=10000` and `CKPT_EVERY=2000`.
Override those at submission time if this should become a longer or shorter
run set, for example:

```bash
N_TRAINING_STEPS=30000 CKPT_EVERY=5000 sbatch jobs/job_sets/flowers64_n_sweep_20260425/submit.sh
```
