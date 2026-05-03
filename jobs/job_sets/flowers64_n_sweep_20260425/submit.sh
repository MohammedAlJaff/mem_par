#!/bin/bash
#SBATCH -A berzelius-2026-78
#SBATCH -J mp_flw64_n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --chdir=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/flowers64_n_sweep_20260425/slurm/%x_%A_%a.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/flowers64_n_sweep_20260425/slurm/%x_%A_%a.err
#SBATCH --array=0-17%3

set -euo pipefail

DEFAULT_REPO_ROOT="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par"
REPO_ROOT="${MP_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
if [[ ! -d "${REPO_ROOT}/src/mp_runners" ]]; then
  echo "REPO_ROOT=${REPO_ROOT} does not look like the repository root." >&2
  echo "Set MP_REPO_ROOT to the mem_par repository root." >&2
  exit 2
fi
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/jobs/job_sets/flowers64_n_sweep_20260425"

module load Mambaforge
mamba activate mem_param_env

export PYTHONPATH=src:.

JOB_SET_NAME="flowers64_n_sweep_20260425"
JOB_SET_DIR="outputs/job_sets/${JOB_SET_NAME}"
WANDB_PROJECT="memparam-${JOB_SET_NAME}"
WANDB_GROUP="${JOB_SET_NAME}"

DATASET="flowers"
IMAGE_SIZE=64
PARAMS=(x eps v)
N_TRAINS=(128 256 512 1024 2048 4096)

N_TEST="${N_TEST:-1020}"
SEED="${SEED:-0}"
N_TRAINING_STEPS="${N_TRAINING_STEPS:-10000}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
BATCH_CAP="${BATCH_CAP:-32}"
MONITOR_BATCH_SIZE="${MONITOR_BATCH_SIZE:-128}"
GENERATION_MONITOR_SAMPLES="${GENERATION_MONITOR_SAMPLES:-8}"
RANDOM_PROJECTED_F_MEM_SAMPLES="${RANDOM_PROJECTED_F_MEM_SAMPLES:-128}"
SAMPLING_STEPS="${SAMPLING_STEPS:-16}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
NUM_PARAMS="${#PARAMS[@]}"
N_IDX=$((TASK_ID / NUM_PARAMS))
PARAM_IDX=$((TASK_ID % NUM_PARAMS))

if (( N_IDX < 0 || N_IDX >= ${#N_TRAINS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
  exit 2
fi

N_TRAIN="${N_TRAINS[$N_IDX]}"
PRED="${PARAMS[$PARAM_IDX]}"
LOSS="${PRED}"
DEFAULT_EVAL_SAMPLE_BANK_SIZE=$(( N_TRAIN > 1024 ? N_TRAIN : 1024 ))
EVAL_SAMPLE_BANK_SIZE="${EVAL_SAMPLE_BANK_SIZE:-${DEFAULT_EVAL_SAMPLE_BANK_SIZE}}"
RUN_NAME="${JOB_SET_NAME}_${DATASET}${IMAGE_SIZE}_${PRED}_${LOSS}_N${N_TRAIN}_s${SEED}"

mkdir -p "${JOB_SET_DIR}" "${SCRIPT_DIR}/slurm"

echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "job_set=${JOB_SET_NAME}"
echo "out_dir=${JOB_SET_DIR}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_group=${WANDB_GROUP}"
echo "run_name=${RUN_NAME}"
echo "dataset=${DATASET}"
echo "image_size=${IMAGE_SIZE}"
echo "prediction_space=${PRED}"
echo "loss_space=${LOSS}"
echo "n_train=${N_TRAIN}"
echo "n_test=${N_TEST}"
echo "eval_sample_bank_size=${EVAL_SAMPLE_BANK_SIZE}"
echo "n_training_steps=${N_TRAINING_STEPS}"
echo "ckpt_every=${CKPT_EVERY}"
echo "seed=${SEED}"

python -m mp_runners.image_train_runner \
  --device cuda \
  --out_dir "${JOB_SET_DIR}" \
  --run_name "${RUN_NAME}" \
  --dataset "${DATASET}" \
  --image_size "${IMAGE_SIZE}" \
  --model unet \
  --prediction_space "${PRED}" \
  --loss_space "${LOSS}" \
  --seed "${SEED}" \
  --n_train "${N_TRAIN}" \
  --n_test "${N_TEST}" \
  --n_training_steps "${N_TRAINING_STEPS}" \
  --ckpt_every "${CKPT_EVERY}" \
  --batch_cap "${BATCH_CAP}" \
  --monitor_batch_size "${MONITOR_BATCH_SIZE}" \
  --generation_monitor \
  --generation_monitor_samples "${GENERATION_MONITOR_SAMPLES}" \
  --no_generation_monitor_show \
  --eval_sample_bank_size "${EVAL_SAMPLE_BANK_SIZE}" \
  --random_projected_f_mem_samples "${RANDOM_PROJECTED_F_MEM_SAMPLES}" \
  --random_projected_f_mem_resize 16 \
  --random_projected_f_mem_proj_dim 256 \
  --sampling_steps "${SAMPLING_STEPS}" \
  --precision tf32 \
  --wandb \
  --wandb_mode online \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_group "${WANDB_GROUP}"
