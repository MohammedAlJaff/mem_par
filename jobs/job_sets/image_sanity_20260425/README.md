# Image Sanity Job set 2026-04-25

Purpose: quick train-only GPU checks for the current image runner API before
larger experiments.

Array:

- datasets: `flowers --image_size 64`, `celeba --image_size 32`, `mnist --image_size 32`
- objectives: `x/x`, `eps/eps`, `v/v`
- `n_train=128`, `n_test=128`, `seed=0`
- `n_training_steps=100`, `ckpt_every=50`
- `random_projected_f_mem_samples=128`, `monitor_batch_size=128`

Submit:

```bash
sbatch jobs/job_sets/image_sanity_20260425/submit.sh
```

Debug one run:

```bash
bash jobs/job_sets/image_sanity_20260425/debug_one.sh flowers v 0
```

Outputs:

- runner attempts: `outputs/job_sets/image_sanity_20260425/runs/`
- Slurm logs: `jobs/job_sets/image_sanity_20260425/slurm/`
- W&B project: `memparam-image_sanity_20260425`
- W&B group: `image_sanity_20260425`
