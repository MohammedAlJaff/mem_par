#!/bin/bash
set -euo pipefail

# Debug one representative Flowers102 64x64 image run before submitting the
# full array.
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
# Flowers102 uses:
#   train split     : official test split, 6149 images
#   reference split : official train split, 1020 images
#
# The reference split is small, so n_eval is fixed at 1020. Generated
# eval_samples still grows with N in the full sweep so high-N runs have more
# chances to expose close train-neighbor samples.
#
# Expected W&B:
# project: memparam-flowers102_64-unet
# name contains: flowers102_64-unet-N512-B32-eps_eps-steps10000-s0

module load Mambaforge
mamba activate mem_param_env

export PYTHONPATH=src:.

python -m mp_runners.image_runner \
  --device cuda \
  --dataset flowers102 \
  --image-size 64 \
  --model unet \
  --n-train 512 \
  --n-eval 1020 \
  --eval-samples 1020 \
  --prediction-space eps \
  --loss-space eps \
  --seed 0 \
  --n-steps 10000 \
  --batch-cap 32 \
  --precision tf32 \
  --sampling-steps 24 \
  --ckpt-every 2000 \
  --final-eval \
  --generation-trace \
  --no-generation-trace-show \
  --wandb \
  --wandb-mode online
