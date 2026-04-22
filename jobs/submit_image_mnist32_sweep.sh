#!/bin/bash
#SBATCH -A YOUR_PROJECT_ACCOUNT
#SBATCH -J mp_mnist32_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -C thin
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-62

set -euo pipefail

# MNIST32 image memorization sweep.
#
# Each SLURM array task is one independent run:
# - one dataset size N
# - one seed
# - one diagonal objective pair
# - one thin node
# - one Python process
# - one GPU
#
# Sweep axes:
#   objective in {x/x, eps/eps, v/v}
#   N in {128, 256, 512, 1024, 2048, 4096, 8192}
#   seed in {0, 1, 2}
#
# Total jobs:
#   3 objectives x 7 N values x 3 seeds = 63 jobs
#
# Data/reference rule, checked with the MNIST loader:
#   MNIST train split     : 60000 images
#   MNIST reference split : official test split
#   balanced n_eval cap   : 8920
#
# The cap is 8920, not 10000, because balanced reference selection needs the
# same number of examples from each digit and digit 5 has only 892 test images.
#
# Evaluation sample rule:
#   eval_samples = max(4096, N)
#   n_eval       = eval_samples
#
# The largest N is 8192 so eval_samples remains below the balanced reference
# cap and every run has at least 1x N generated samples.
#
# W&B:
#   Do not pass --wandb-name, --wandb-project, or --wandb-group.
#   The runner will generate project/name/group from the final config.
#
# Checkpointing:
#   bonnaire-1m gives dense early checkpoints and coarser later checkpoints.
#   It requires n_steps <= 1000000.

module load Mambaforge
mamba activate mem_param_env

export PYTHONPATH=src:.

PARAMS=(x eps v)
NTRAINS=(128 256 512 1024 2048 4096 8192)
SEEDS=(0 1 2)

NUM_N=${#NTRAINS[@]}
NUM_SEEDS=${#SEEDS[@]}

P_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_N * NUM_SEEDS)))
REM=$((SLURM_ARRAY_TASK_ID % (NUM_N * NUM_SEEDS)))
N_IDX=$((REM / NUM_SEEDS))
SEED_IDX=$((REM % NUM_SEEDS))

PRED="${PARAMS[$P_IDX]}"
LOSS="${PARAMS[$P_IDX]}"
N_TRAIN="${NTRAINS[$N_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

EVAL_FLOOR=4096
EVAL_SAMPLES=$(( N_TRAIN > EVAL_FLOOR ? N_TRAIN : EVAL_FLOOR ))
N_EVAL="${EVAL_SAMPLES}"

BATCH_CAP=64
N_STEPS=1000000

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "dataset=mnist"
echo "image_size=32"
echo "prediction_space=${PRED}"
echo "loss_space=${LOSS}"
echo "n_train=${N_TRAIN}"
echo "n_eval=${N_EVAL}"
echo "eval_samples=${EVAL_SAMPLES}"
echo "seed=${SEED}"

python -m mp_runners.image_runner \
  --device cuda \
  --dataset mnist \
  --image-size 32 \
  --model unet \
  --n-train "${N_TRAIN}" \
  --n-eval "${N_EVAL}" \
  --eval-samples "${EVAL_SAMPLES}" \
  --prediction-space "${PRED}" \
  --loss-space "${LOSS}" \
  --seed "${SEED}" \
  --n-steps "${N_STEPS}" \
  --batch-cap "${BATCH_CAP}" \
  --precision tf32 \
  --sampling-steps 24 \
  --ckpt-schedule bonnaire-1m \
  --final-eval \
  --generation-trace \
  --no-generation-trace-show \
  --wandb \
  --wandb-mode online
