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
SCRIPT_DIR="${REPO_ROOT}/jobs/job_sets/synthetic_sanity_20260425"

module load Mambaforge
mamba activate mem_param_env

export PYTHONPATH=src:.

DATASET="${1:-gmm_subspace}"
OBJECTIVE="${2:-v}"
SEED="${3:-0}"

case "${DATASET}" in
  gmm_subspace)
    AMBIENT_DIM=2
    INTRINSIC_DATA_DIM=2
    ;;
  torus3d)
    AMBIENT_DIM=3
    INTRINSIC_DATA_DIM=""
    ;;
  *)
    echo "Unsupported synthetic sanity dataset: ${DATASET}" >&2
    echo "Use one of: gmm_subspace, torus3d" >&2
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

JOB_SET_NAME="synthetic_sanity_20260425"
JOB_SET_DIR="outputs/job_sets/${JOB_SET_NAME}"
WANDB_PROJECT="memparam-${JOB_SET_NAME}"
WANDB_GROUP="${JOB_SET_NAME}"
RUN_NAME="${JOB_SET_NAME}_${DATASET}_${OBJECTIVE}_${OBJECTIVE}_N128_s${SEED}_debug"

mkdir -p "${JOB_SET_DIR}" "${SCRIPT_DIR}/slurm"

echo "job_set=${JOB_SET_NAME}"
echo "out_dir=${JOB_SET_DIR}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_group=${WANDB_GROUP}"
echo "run_name=${RUN_NAME}"
echo "dataset=${DATASET}"
echo "ambient_dim=${AMBIENT_DIM}"
echo "prediction_space=${OBJECTIVE}"
echo "loss_space=${OBJECTIVE}"
echo "seed=${SEED}"

CMD=(
  python -m mp_runners.synthetic_train_runner
  --device cuda
  --out_dir "${JOB_SET_DIR}"
  --run_name "${RUN_NAME}"
  --dataset "${DATASET}"
  --ambient_dim "${AMBIENT_DIM}"
  --prediction_space "${OBJECTIVE}"
  --loss_space "${OBJECTIVE}"
  --seed "${SEED}"
  --n_train 128
  --n_test 1024
  --n_training_steps 100
  --ckpt_every 50
  --batch_cap 128
  --lr 1e-3
  --eval_sample_bank_size 1024
  --sampling_method heun
  --sampling_steps 32
  --wandb
  --wandb_mode online
  --wandb_project "${WANDB_PROJECT}"
  --wandb_group "${WANDB_GROUP}"
)

if [[ -n "${INTRINSIC_DATA_DIM}" ]]; then
  CMD+=(--intrinsic_data_dim "${INTRINSIC_DATA_DIM}")
fi

"${CMD[@]}"
