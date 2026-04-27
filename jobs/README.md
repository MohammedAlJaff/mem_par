Run job sets from the repository root with the current train-only runner APIs.

Current job sets:

- `job_sets/image_sanity_20260425`: 9 image smoke jobs covering
  Flowers64, CelebA32, MNIST32 and the diagonal `x/x`, `eps/eps`, `v/v`
  parameterizations.
- `job_sets/synthetic_sanity_20260425`: 6 synthetic smoke jobs covering
  GMM-subspace, torus3d and the same diagonal parameterizations.
- `job_sets/flowers64_n_sweep_20260425`: 18 Flowers64 jobs fixing the
  dataset while sweeping `N_train` and diagonal `x/x`, `eps/eps`, `v/v`
  parameterizations.
- `job_sets/mem_gen_boundary_20260427`: 84 CelebA32 grayscale jobs for the
  memorization/generalization boundary sweep over `N_train`, three seeds,
  diagonal parameterizations, and `x/v`.

Submit an array:

```bash
sbatch jobs/job_sets/image_sanity_20260425/submit.sh
sbatch jobs/job_sets/synthetic_sanity_20260425/submit.sh
sbatch jobs/job_sets/flowers64_n_sweep_20260425/submit.sh
sbatch jobs/job_sets/mem_gen_boundary_20260427/submit.sh
```

Run one debug job interactively:

```bash
bash jobs/job_sets/image_sanity_20260425/debug_one.sh flowers v 0
bash jobs/job_sets/synthetic_sanity_20260425/debug_one.sh gmm_subspace v 0
bash jobs/job_sets/flowers64_n_sweep_20260425/debug_one.sh 512 v 0
```

Outputs go under `outputs/job_sets/<job_set_name>/runs/`. Slurm stdout/stderr
go under each job set's `slurm/` directory. Each job set also sets an
explicit W&B project, `memparam-<job_set_name>`, so all runs from the same
job set land together online.
