#!/bin/bash
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

DATASET="${1:-flowers}"
OBJECTIVE="${2:-v}"
SEED="${3:-0}"

case "${DATASET}" in
  flowers)
    IMAGE_SIZE=64
    ;;
  celeba|mnist)
    IMAGE_SIZE=32
    ;;
  *)
    echo "Unsupported image sanity dataset: ${DATASET}" >&2
    echo "Use one of: flowers, celeba, mnist" >&2
    exit 2
    ;;
esac

case "${OBJECTIVE}" in
  x|eps|v)
    ;;
  *)
    echo "Unsupported objective: ${OBJECTIVE}" >&2
    echo "Use one of: x, eps, v" >&2
    exit 2
    ;;
esac

JOB_SET_NAME="image_sanity_20260425"
JOB_SET_DIR="outputs/job_sets/${JOB_SET_NAME}"
WANDB_PROJECT="memparam-${JOB_SET_NAME}"
WANDB_GROUP="${JOB_SET_NAME}"
RUN_NAME="${JOB_SET_NAME}_${DATASET}${IMAGE_SIZE}_${OBJECTIVE}_${OBJECTIVE}_N128_s${SEED}_debug"

mkdir -p "${JOB_SET_DIR}" "${SCRIPT_DIR}/slurm"

echo "job_set=${JOB_SET_NAME}"
echo "out_dir=${JOB_SET_DIR}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_group=${WANDB_GROUP}"
echo "run_name=${RUN_NAME}"
echo "dataset=${DATASET}"
echo "image_size=${IMAGE_SIZE}"
echo "prediction_space=${OBJECTIVE}"
echo "loss_space=${OBJECTIVE}"
echo "seed=${SEED}"

python -m mp_runners.image_train_runner \
  --device cuda \
  --out_dir "${JOB_SET_DIR}" \
  --run_name "${RUN_NAME}" \
  --dataset "${DATASET}" \
  --image_size "${IMAGE_SIZE}" \
  --model unet \
  --prediction_space "${OBJECTIVE}" \
  --loss_space "${OBJECTIVE}" \
  --seed "${SEED}" \
  --n_train 128 \
  --n_test 128 \
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
