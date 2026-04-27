#!/bin/bash
#SBATCH -A berzelius-2026-78
#SBATCH -J mp_mgb_c32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --chdir=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/mem_gen_boundary_20260427/slurm/%x_%A_%a.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/mem_gen_boundary_20260427/slurm/%x_%A_%a.err
#SBATCH --array=0-83

set -euo pipefail

DEFAULT_REPO_ROOT="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par"
REPO_ROOT="${MP_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
if [[ ! -d "${REPO_ROOT}/src/mp_runners" ]]; then
  echo "REPO_ROOT=${REPO_ROOT} does not look like the repository root." >&2
  echo "Set MP_REPO_ROOT to the mem_par repository root." >&2
  exit 2
fi
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/jobs/job_sets/mem_gen_boundary_20260427"

module load Mambaforge
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
mamba activate mem_param_env

export PYTHONPATH=src:.

JOB_SET_NAME="mem_gen_boundary_20260427"
JOB_SET_DIR="${JOB_SET_DIR:-outputs/job_sets/${JOB_SET_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-memparam-${JOB_SET_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-${JOB_SET_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"

DATASET="celeba"
IMAGE_SIZE=32
N_TRAINS=(256 512 1024 2048 4096 8192 16384)
SEEDS=(0 1 2)
PREDICTION_SPACES=(x eps v x)
LOSS_SPACES=(x eps v v)
PARAM_LABELS=(x_x eps_eps v_v x_v)

N_TEST="${N_TEST:-1024}"
N_TRAINING_STEPS="${N_TRAINING_STEPS:-1000000}"
CKPT_SCHEDULE="${CKPT_SCHEDULE:-bonnaire-1m}"
CKPT_EVERY="${CKPT_EVERY:-5000}"
CKPT_STEPS="${CKPT_STEPS:-}"
BATCH_CAP="${BATCH_CAP:-256}"
MONITOR_BATCH_SIZE="${MONITOR_BATCH_SIZE:-128}"
GENERATION_MONITOR_SAMPLES="${GENERATION_MONITOR_SAMPLES:-8}"
RANDOM_PROJECTED_F_MEM_SAMPLES="${RANDOM_PROJECTED_F_MEM_SAMPLES:-128}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-}"
RESUME_FROM_ATTEMPT="${RESUME_FROM_ATTEMPT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
NUM_PARAMS="${#PREDICTION_SPACES[@]}"
NUM_SEEDS="${#SEEDS[@]}"
RUNS_PER_N=$((NUM_SEEDS * NUM_PARAMS))
N_IDX=$((TASK_ID / RUNS_PER_N))
SEED_PARAM_IDX=$((TASK_ID % RUNS_PER_N))
SEED_IDX=$((SEED_PARAM_IDX / NUM_PARAMS))
PARAM_IDX=$((SEED_PARAM_IDX % NUM_PARAMS))

if (( TASK_ID < 0 || N_IDX >= ${#N_TRAINS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
  exit 2
fi

if [[ -n "${RESUME_FROM_ATTEMPT}" && -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Set only one of RESUME_FROM_ATTEMPT or RESUME_FROM_CHECKPOINT." >&2
  exit 2
fi

N_TRAIN="${N_TRAINS[$N_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
PRED="${PREDICTION_SPACES[$PARAM_IDX]}"
LOSS="${LOSS_SPACES[$PARAM_IDX]}"
PARAM_LABEL="${PARAM_LABELS[$PARAM_IDX]}"
DEFAULT_EVAL_SAMPLE_BANK_SIZE=$(( N_TRAIN > 1024 ? N_TRAIN : 1024 ))
EVAL_SAMPLE_BANK_SIZE="${EVAL_SAMPLE_BANK_SIZE:-${DEFAULT_EVAL_SAMPLE_BANK_SIZE}}"
RUN_NAME="${JOB_SET_NAME}_${DATASET}${IMAGE_SIZE}_N${N_TRAIN}_B${BATCH_CAP}_${PARAM_LABEL}_steps${N_TRAINING_STEPS}_s${SEED}"

mkdir -p "${JOB_SET_DIR}" "${SCRIPT_DIR}/slurm"

echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "job_set=${JOB_SET_NAME}"
echo "out_dir=${JOB_SET_DIR}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_group=${WANDB_GROUP}"
echo "wandb_mode=${WANDB_MODE}"
echo "run_name=${RUN_NAME}"
echo "dataset=${DATASET}"
echo "image_size=${IMAGE_SIZE}"
echo "prediction_space=${PRED}"
echo "loss_space=${LOSS}"
echo "n_train=${N_TRAIN}"
echo "n_test=${N_TEST}"
echo "eval_sample_bank_size=${EVAL_SAMPLE_BANK_SIZE}"
echo "n_training_steps=${N_TRAINING_STEPS}"
echo "ckpt_schedule=${CKPT_SCHEDULE}"
echo "ckpt_every=${CKPT_EVERY}"
echo "ckpt_steps=${CKPT_STEPS}"
echo "batch_cap=${BATCH_CAP}"
echo "seed=${SEED}"
echo "stop_after_step=${STOP_AFTER_STEP}"
echo "resume_from_attempt=${RESUME_FROM_ATTEMPT}"
echo "resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"

CMD=(
  python -m mp_runners.image_train_runner
  --device cuda
  --out_dir "${JOB_SET_DIR}"
  --run_name "${RUN_NAME}"
  --dataset "${DATASET}"
  --image_size "${IMAGE_SIZE}"
  --model unet
  --prediction_space "${PRED}"
  --loss_space "${LOSS}"
  --seed "${SEED}"
  --n_train "${N_TRAIN}"
  --n_test "${N_TEST}"
  --n_training_steps "${N_TRAINING_STEPS}"
  --batch_cap "${BATCH_CAP}"
  --monitor_batch_size "${MONITOR_BATCH_SIZE}"
  --generation_monitor
  --generation_monitor_samples "${GENERATION_MONITOR_SAMPLES}"
  --no_generation_monitor_show
  --eval_sample_bank_size "${EVAL_SAMPLE_BANK_SIZE}"
  --random_projected_f_mem_samples "${RANDOM_PROJECTED_F_MEM_SAMPLES}"
  --random_projected_f_mem_resize 16
  --random_projected_f_mem_proj_dim 256
  --sampling_steps "${SAMPLING_STEPS}"
  --precision tf32
  --wandb
  --wandb_mode "${WANDB_MODE}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${RUN_NAME}"
  --wandb_group "${WANDB_GROUP}"
)

if [[ -n "${CKPT_STEPS}" ]]; then
  CMD+=(--ckpt_steps "${CKPT_STEPS}")
elif [[ "${CKPT_SCHEDULE}" == "interval" ]]; then
  CMD+=(--ckpt_every "${CKPT_EVERY}")
else
  CMD+=(--ckpt_schedule "${CKPT_SCHEDULE}")
fi

if [[ -n "${STOP_AFTER_STEP}" ]]; then
  CMD+=(--stop_after_step "${STOP_AFTER_STEP}")
fi
if [[ -n "${RESUME_FROM_ATTEMPT}" ]]; then
  CMD+=(--resume_from_attempt "${RESUME_FROM_ATTEMPT}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${CMD[@]}"
