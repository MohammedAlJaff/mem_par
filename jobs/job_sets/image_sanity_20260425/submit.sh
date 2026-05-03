#!/bin/bash
#SBATCH -A berzelius-2026-78
#SBATCH -J mp_img_sanity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --chdir=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/image_sanity_20260425/slurm/%x_%A_%a.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/image_sanity_20260425/slurm/%x_%A_%a.err
#SBATCH --array=0-8%3

set -euo pipefail

DEFAULT_REPO_ROOT="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par"
REPO_ROOT="${MP_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
if [[ ! -d "${REPO_ROOT}/src/mp_runners" ]]; then
  echo "REPO_ROOT=${REPO_ROOT} does not look like the repository root." >&2
  echo "Set MP_REPO_ROOT to the mem_par repository root." >&2
  exit 2
fi
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/jobs/job_sets/image_sanity_20260425"

module load Mambaforge
mamba activate mem_param_env

export PYTHONPATH=src:.

JOB_SET_NAME="image_sanity_20260425"
JOB_SET_DIR="outputs/job_sets/${JOB_SET_NAME}"
WANDB_PROJECT="memparam-${JOB_SET_NAME}"
WANDB_GROUP="${JOB_SET_NAME}"

DATASETS=(flowers celeba mnist)
IMAGE_SIZES=(64 32 32)
PARAMS=(x eps v)

N_TRAIN=128
N_TEST=128
SEED=0

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
NUM_PARAMS="${#PARAMS[@]}"
DATASET_IDX=$((TASK_ID / NUM_PARAMS))
PARAM_IDX=$((TASK_ID % NUM_PARAMS))

if (( DATASET_IDX < 0 || DATASET_IDX >= ${#DATASETS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
  exit 2
fi

DATASET="${DATASETS[$DATASET_IDX]}"
IMAGE_SIZE="${IMAGE_SIZES[$DATASET_IDX]}"
PRED="${PARAMS[$PARAM_IDX]}"
LOSS="${PRED}"
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
  --n_training_steps 100 \
  --ckpt_every 50 \
  --batch_cap 32 \
  --monitor_batch_size 128 \
  --generation_monitor \
  --generation_monitor_samples 8 \
  --no_generation_monitor_show \
  --eval_sample_bank_size 256 \
  --random_projected_f_mem_samples 128 \
  --random_projected_f_mem_resize 16 \
  --random_projected_f_mem_proj_dim 256 \
  --sampling_steps 16 \
  --precision tf32 \
  --wandb \
  --wandb_mode online \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_group "${WANDB_GROUP}"
