# Synthetic Sanity Job set 2026-04-25

Purpose: quick train-only GPU checks for the current synthetic runner API.

Array:

- datasets: `gmm_subspace`, `torus3d`
- objectives: `x/x`, `eps/eps`, `v/v`
- `n_train=128`, `n_test=1024`, `seed=0`
- `n_training_steps=100`, `ckpt_every=50`

Submit:

```bash
sbatch jobs/job_sets/synthetic_sanity_20260425/submit.sh
```

Debug one run:

```bash
bash jobs/job_sets/synthetic_sanity_20260425/debug_one.sh gmm_subspace v 0
```

Outputs:

- runner attempts: `outputs/job_sets/synthetic_sanity_20260425/runs/`
- Slurm logs: `jobs/job_sets/synthetic_sanity_20260425/slurm/`
- W&B project: `memparam-synthetic_sanity_20260425`
- W&B group: `synthetic_sanity_20260425`
