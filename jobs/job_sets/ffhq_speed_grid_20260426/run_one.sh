#!/bin/bash
# Shared body for strict thin/fat FFHQ speed-grid Slurm array tasks.

set -euo pipefail

DEFAULT_REPO_ROOT="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par"
REPO_ROOT="${MP_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
if [[ ! -d "${REPO_ROOT}/src/mp_runners" ]]; then
  echo "REPO_ROOT=${REPO_ROOT} does not look like the repository root." >&2
  echo "Set MP_REPO_ROOT to the mem_par repository root." >&2
  exit 2
fi
cd "${REPO_ROOT}"

NODE_CLASS="${NODE_CLASS:?NODE_CLASS must be set to thin or fat}"
case "${NODE_CLASS}" in
  thin|fat)
    ;;
  *)
    echo "Unsupported NODE_CLASS=${NODE_CLASS}; use thin or fat." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="${REPO_ROOT}/jobs/job_sets/ffhq_speed_grid_20260426"
RESULTS_DIR="${SCRIPT_DIR}/results_${NODE_CLASS}"
mkdir -p "${SCRIPT_DIR}/slurm_${NODE_CLASS}" "${RESULTS_DIR}"

module load Mambaforge
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
mamba activate mem_param_env
export PYTHONPATH=src:.

JOB_SET_NAME="ffhq_speed_grid_20260426_${NODE_CLASS}"
JOB_SET_DIR="outputs/job_sets/${JOB_SET_NAME}"
DATASET="${DATASET:-ffhq}"
OBJECTIVE="${OBJECTIVE:-v}"
SEED="${SEED:-0}"
N_TRAIN="${N_TRAIN:-4000}"
N_TEST="${N_TEST:-1024}"
N_STEPS="${N_STEPS:-1000}"
EVAL_SAMPLE_BANK_SIZE="${EVAL_SAMPLE_BANK_SIZE:-1024}"
MONITOR_BATCH_SIZE="${MONITOR_BATCH_SIZE:-128}"
SAMPLING_STEPS="${SAMPLING_STEPS:-8}"
RANDOM_PROJECTED_F_MEM_SAMPLES="${RANDOM_PROJECTED_F_MEM_SAMPLES:-1}"

IMAGE_SIZES=(32 64 256 512)
BATCH_CAPS=(16 32 64 128 256 512)
SUPPORTED_IMAGE_SIZES=(32 64 128 256)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
NUM_BATCHES="${#BATCH_CAPS[@]}"
IMAGE_IDX=$(( TASK_ID / NUM_BATCHES ))
BATCH_IDX=$(( TASK_ID % NUM_BATCHES ))

if (( IMAGE_IDX < 0 || IMAGE_IDX >= ${#IMAGE_SIZES[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
  exit 2
fi

IMAGE_SIZE="${IMAGE_SIZES[$IMAGE_IDX]}"
BATCH_CAP="${BATCH_CAPS[$BATCH_IDX]}"
RUN_NAME="${JOB_SET_NAME}_${DATASET}${IMAGE_SIZE}_B${BATCH_CAP}_${OBJECTIVE}_${OBJECTIVE}_N${N_TRAIN}_s${SEED}"
RESULT_PATH="${RESULTS_DIR}/image${IMAGE_SIZE}_batch${BATCH_CAP}.tsv"

write_result() {
  local status="$1"
  local train_elapsed_sec="$2"
  local steps_per_sec="$3"
  local status_elapsed_sec="$4"
  local status_path="$5"

  {
    printf "node_class\tdataset\timage_size\tbatch_cap\tn_steps\tobjective\tseed\tstatus\ttrain_elapsed_sec\tsteps_per_sec\tstatus_elapsed_sec\trun_name\tstatus_path\n"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${NODE_CLASS}" \
      "${DATASET}" \
      "${IMAGE_SIZE}" \
      "${BATCH_CAP}" \
      "${N_STEPS}" \
      "${OBJECTIVE}" \
      "${SEED}" \
      "${status}" \
      "${train_elapsed_sec}" \
      "${steps_per_sec}" \
      "${status_elapsed_sec}" \
      "${RUN_NAME}" \
      "${status_path}"
  } > "${RESULT_PATH}"
}

is_supported_image_size() {
  local requested="$1"
  local supported
  for supported in "${SUPPORTED_IMAGE_SIZES[@]}"; do
    if [[ "${requested}" == "${supported}" ]]; then
      return 0
    fi
  done
  return 1
}

if ! is_supported_image_size "${IMAGE_SIZE}"; then
  echo "Skipping image_size=${IMAGE_SIZE}: runner supports ${SUPPORTED_IMAGE_SIZES[*]}."
  write_result "skipped_unsupported_image_size" "" "" "" ""
  exit 0
fi

mkdir -p "${JOB_SET_DIR}"

echo "job_set=${JOB_SET_NAME}"
echo "node_class=${NODE_CLASS}"
echo "dataset=${DATASET}"
echo "image_size=${IMAGE_SIZE}"
echo "batch_cap=${BATCH_CAP}"
echo "n_steps=${N_STEPS}"
echo "objective=${OBJECTIVE}"
echo "seed=${SEED}"
echo "run_name=${RUN_NAME}"
echo "result_path=${RESULT_PATH}"
write_result "started" "" "" "" ""

CMD=(
  python -m mp_runners.image_train_runner
  --device cuda
  --out_dir "${JOB_SET_DIR}"
  --run_name "${RUN_NAME}"
  --dataset "${DATASET}"
  --image_size "${IMAGE_SIZE}"
  --model unet
  --prediction_space "${OBJECTIVE}"
  --loss_space "${OBJECTIVE}"
  --seed "${SEED}"
  --n_train "${N_TRAIN}"
  --n_test "${N_TEST}"
  --n_training_steps "${N_STEPS}"
  --ckpt_steps "${N_STEPS}"
  --batch_cap "${BATCH_CAP}"
  --monitor_batch_size "${MONITOR_BATCH_SIZE}"
  --generation_monitor_samples 1
  --eval_sample_bank_size "${EVAL_SAMPLE_BANK_SIZE}"
  --random_projected_f_mem_samples "${RANDOM_PROJECTED_F_MEM_SAMPLES}"
  --random_projected_f_mem_resize 16
  --random_projected_f_mem_proj_dim 256
  --sampling_steps "${SAMPLING_STEPS}"
  --precision tf32
  --no_wandb
)

set +e
"${CMD[@]}"
exit_code=$?
set -e

shopt -s nullglob
status_paths=("${JOB_SET_DIR}/runs/${RUN_NAME}-"*/seeds/seed_*/attempts/*/status.json)
shopt -u nullglob

if (( ${#status_paths[@]} == 0 )); then
  write_result "failed_exit_${exit_code}" "" "" "" ""
  exit 0
fi

latest_status="$(
  ls -t "${status_paths[@]}" 2>/dev/null | head -n 1
)"

python - "${latest_status}" "${exit_code}" "${RESULT_PATH}" <<'PY'
import json
import re
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
exit_code = int(sys.argv[2])
result_path = Path(sys.argv[3])
status = json.loads(status_path.read_text())

progress = status.get("progress") or {}
status_elapsed = progress.get("elapsed_sec", "")
steps_per_sec = progress.get("steps_per_sec", "")

train_elapsed = ""
attempt_log = Path(status.get("attempt_log_path") or "")
if attempt_log.exists():
    pattern = re.compile(r"train step=\d+/\d+.*?elapsed=(\d+):(\d+):(\d+)")
    for line in attempt_log.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            h, m, s = (int(part) for part in match.groups())
            train_elapsed = str(h * 3600 + m * 60 + s)

label = "completed" if exit_code == 0 else f"failed_exit_{exit_code}"
lines = result_path.read_text().splitlines()
header = lines[0].split("\t")
values = lines[1].split("\t")
row = dict(zip(header, values))
row["status"] = label
row["train_elapsed_sec"] = str(train_elapsed)
row["steps_per_sec"] = str(steps_per_sec)
row["status_elapsed_sec"] = str(status_elapsed)
row["status_path"] = str(status_path)
result_path.write_text(
    "\t".join(header) + "\n"
    + "\t".join(str(row.get(key, "")) for key in header) + "\n"
)
PY

echo "wrote ${RESULT_PATH}"
cat "${RESULT_PATH}"
