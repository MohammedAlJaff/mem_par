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
SCRIPT_DIR="${REPO_ROOT}/jobs/job_sets/flowers64_n_sweep_20260425"

module load Mambaforge
mamba activate mem_param_env

export PYTHONPATH=src:.

N_TRAIN="${1:-512}"
OBJECTIVE="${2:-v}"
SEED="${3:-0}"

case "${N_TRAIN}" in
  128|256|512|1024|2048|4096)
    ;;
  *)
    echo "Unsupported N_train for this set: ${N_TRAIN}" >&2
    echo "Use one of: 128, 256, 512, 1024, 2048, 4096" >&2
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

JOB_SET_NAME="flowers64_n_sweep_20260425"
JOB_SET_DIR="outputs/job_sets/${JOB_SET_NAME}"
WANDB_PROJECT="memparam-${JOB_SET_NAME}"
WANDB_GROUP="${JOB_SET_NAME}"
DATASET="flowers"
IMAGE_SIZE=64
N_TEST="${N_TEST:-1020}"
DEFAULT_EVAL_SAMPLE_BANK_SIZE=$(( N_TRAIN > 1024 ? N_TRAIN : 1024 ))
EVAL_SAMPLE_BANK_SIZE="${EVAL_SAMPLE_BANK_SIZE:-${DEFAULT_EVAL_SAMPLE_BANK_SIZE}}"
N_TRAINING_STEPS="${N_TRAINING_STEPS:-200}"
CKPT_EVERY="${CKPT_EVERY:-100}"
BATCH_CAP="${BATCH_CAP:-32}"
MONITOR_BATCH_SIZE="${MONITOR_BATCH_SIZE:-128}"
GENERATION_MONITOR_SAMPLES="${GENERATION_MONITOR_SAMPLES:-8}"
RANDOM_PROJECTED_F_MEM_SAMPLES="${RANDOM_PROJECTED_F_MEM_SAMPLES:-128}"
SAMPLING_STEPS="${SAMPLING_STEPS:-16}"
RUN_NAME="${JOB_SET_NAME}_${DATASET}${IMAGE_SIZE}_${OBJECTIVE}_${OBJECTIVE}_N${N_TRAIN}_s${SEED}_debug"

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
  --prediction_space "${OBJECTIVE}" \
  --loss_space "${OBJECTIVE}" \
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
