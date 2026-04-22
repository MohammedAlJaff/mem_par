#!/bin/bash
set -euo pipefail

# Debug one representative run before submitting the full SLURM array.
# This checks:
# - the Python environment
# - synthetic runner CLI flags
# - W&B logging
# - final evaluation
# - checkpoint replay
#
# Expected W&B:
# project: memparam-gmm_subspace-mlp
# name contains: gmm_subspace-mlp-N1024-B512-eps_eps-steps10000-s0

module load Mambaforge
mamba activate mem_param_env

export PYTHONPATH=src:.

python -m mp_runners.synthetic_runner \
  --device cuda \
  --dataset gmm_subspace \
  --ambient-dim 2 \
  --intrinsic-data-dim 2 \
  --n-train 1024 \
  --n-reference 16384 \
  --eval-samples 16384 \
  --prediction-space eps \
  --loss-space eps \
  --seed 0 \
  --n-steps 10000 \
  --batch-cap 512 \
  --lr 1e-3 \
  --sampling-method heun \
  --sampling-steps 128 \
  --ckpt-every 2000 \
  --final-eval \
  --wandb \
  --wandb-mode online
