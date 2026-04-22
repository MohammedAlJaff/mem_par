#!/bin/bash
set -euo pipefail

# Debug one representative MNIST32 image run before submitting the full array.
# This checks:
# - the Python environment
# - image runner CLI flags
# - W&B automatic project/name/group
# - final evaluation
# - checkpoint replay
# - generation-trace image logging
#
# Data choice checked with:
#   python -m mp_runners.image_runner ... --data-summary
#
# MNIST uses balanced train/reference class selection. The official test split
# has only 892 images for digit 5, so the largest balanced reference set is:
#   n_eval = 892 * 10 = 8920
#
# Expected W&B:
# project: memparam-mnist-unet
# name contains: mnist-unet-N1024-B64-eps_eps-steps10000-s0

module load Mambaforge
mamba activate mem_param_env

export PYTHONPATH=src:.

python -m mp_runners.image_runner \
  --device cuda \
  --dataset mnist \
  --image-size 32 \
  --model unet \
  --n-train 1024 \
  --n-eval 4096 \
  --eval-samples 4096 \
  --prediction-space eps \
  --loss-space eps \
  --seed 0 \
  --n-steps 10000 \
  --batch-cap 64 \
  --precision tf32 \
  --sampling-steps 24 \
  --ckpt-every 2000 \
  --final-eval \
  --generation-trace \
  --no-generation-trace-show \
  --wandb \
  --wandb-mode online
