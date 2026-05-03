#!/bin/bash
#SBATCH -A berzelius-2026-78
#SBATCH -J mp_synth_sanity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/synthetic_sanity_20260425/slurm/%x_%A_%a.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/synthetic_sanity_20260425/slurm/%x_%A_%a.err
#SBATCH --array=0-5%3

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

JOB_SET_NAME="synthetic_sanity_20260425"
JOB_SET_DIR="outputs/job_sets/${JOB_SET_NAME}"
WANDB_PROJECT="memparam-${JOB_SET_NAME}"
WANDB_GROUP="${JOB_SET_NAME}"

DATASETS=(gmm_subspace torus3d)
AMBIENT_DIMS=(2 3)
INTRINSIC_DIMS=(2 "")
PARAMS=(x eps v)

N_TRAIN=128
N_TEST=1024
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
AMBIENT_DIM="${AMBIENT_DIMS[$DATASET_IDX]}"
INTRINSIC_DATA_DIM="${INTRINSIC_DIMS[$DATASET_IDX]}"
PRED="${PARAMS[$PARAM_IDX]}"
LOSS="${PRED}"
RUN_NAME="${JOB_SET_NAME}_${DATASET}_${PRED}_${LOSS}_N${N_TRAIN}_s${SEED}"

mkdir -p "${JOB_SET_DIR}" "${SCRIPT_DIR}/slurm"

echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "job_set=${JOB_SET_NAME}"
echo "out_dir=${JOB_SET_DIR}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_group=${WANDB_GROUP}"
echo "run_name=${RUN_NAME}"
echo "dataset=${DATASET}"
echo "ambient_dim=${AMBIENT_DIM}"
echo "intrinsic_data_dim=${INTRINSIC_DATA_DIM:-NA}"
echo "prediction_space=${PRED}"
echo "loss_space=${LOSS}"
echo "n_train=${N_TRAIN}"
echo "n_test=${N_TEST}"
echo "seed=${SEED}"

CMD=(
  python -m mp_runners.synthetic_train_runner
  --device cuda
  --out_dir "${JOB_SET_DIR}"
  --run_name "${RUN_NAME}"
  --dataset "${DATASET}"
  --ambient_dim "${AMBIENT_DIM}"
  --prediction_space "${PRED}"
  --loss_space "${LOSS}"
  --seed "${SEED}"
  --n_train "${N_TRAIN}"
  --n_test "${N_TEST}"
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
