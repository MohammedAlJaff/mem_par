#!/bin/bash
#SBATCH -A berzelius-2026-78
#SBATCH -J mp_gmm_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-11

set -euo pipefail

# Synthetic GMM memorization sweep.
#
# Each SLURM array task is one independent run:
# - one dataset size N
# - one seed
# - one diagonal objective pair
# - one node
# - one Python process
# - one GPU
#
# Sweep axes:
#   objective in {x/x, eps/eps, v/v}
#   N in {128, 256, 512, 1024, 2048, 4096, 8192, 16384}
#   seed in {0, 1, 2}
#
# Total jobs:
#   3 objectives x 8 N values x 3 seeds = 72 jobs
#
# Evaluation sample rule:
#   eval_samples = max(16384, N)
#   n_reference  = eval_samples
#
# This gives much more generated-sample coverage at low N:
#   N=128     eval_samples=16384   128x N
#   N=256     eval_samples=16384    64x N
#   N=512     eval_samples=16384    32x N
#   N=1024    eval_samples=16384    16x N
#   N=2048    eval_samples=16384     8x N
#   N=4096    eval_samples=16384     4x N
#   N=8192    eval_samples=16384     2x N
#   N=16384   eval_samples=16384     1x N
#
# W&B:
#   Do not pass --wandb-name or --wandb-project.
#   The runner will generate project/name/group from the final config.
#
# Checkpointing:
#   bonnaire-1m gives dense early checkpoints and coarser later checkpoints.
#   It requires n_steps <= 1000000.

module load Mambaforge
mamba activate mem_param_env

export PYTHONPATH=src:.

PARAMS=(x eps v)
NTRAINS=(128 256 512 1024 )
SEEDS=(12345)

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

EVAL_SAMPLES=$(( N_TRAIN > 16384 ? N_TRAIN : 16384 ))
N_EVAL="${EVAL_SAMPLES}"

BATCH_CAP=512
N_STEPS=1000000

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "prediction_space=${PRED}"
echo "loss_space=${LOSS}"
echo "n_train=${N_TRAIN}"
echo "n_reference=${N_EVAL}"
echo "eval_samples=${EVAL_SAMPLES}"
echo "seed=${SEED}"

python -m mp_runners.synthetic_runner \
  --device cuda \
  --dataset gmm_subspace \
  --ambient-dim 2 \
  --intrinsic-data-dim 2 \
  --n-train "${N_TRAIN}" \
  --n-reference "${N_EVAL}" \
  --eval-samples "${EVAL_SAMPLES}" \
  --prediction-space "${PRED}" \
  --loss-space "${LOSS}" \
  --seed "${SEED}" \
  --n-steps "${N_STEPS}" \
  --batch-cap "${BATCH_CAP}" \
  --lr 1e-3 \
  --sampling-method heun \
  --sampling-steps 128 \
  --ckpt-schedule bonnaire-1m \
  --final-eval \
  --wandb \
  --wandb-mode online
