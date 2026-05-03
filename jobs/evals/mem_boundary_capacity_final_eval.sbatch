#!/bin/bash
#SBATCH -A berzelius-2026-78
#SBATCH -J mp_mgb_cap_final_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-96
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/evals/logs/%x_%A_%a.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/evals/logs/%x_%A_%a.err

set -euo pipefail

# Full paper-quality final-checkpoint eval for completed capacity runs.
# This intentionally uses the frozen manifest written before submission, so
# later-finishing resume attempts cannot shift SLURM_ARRAY_TASK_ID mappings.
REPO_ROOT="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par"
OUT_DIR="${OUT_DIR:-outputs/evals/mem_boundary_capacity_20260427_final_quality_20260503}"
MANIFEST_CSV="${MANIFEST_CSV:-${OUT_DIR}/manifest.csv}"

cd "$REPO_ROOT"
mkdir -p jobs/evals/logs

module load Mambaforge
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
mamba activate mem_param_env

export PYTHONPATH=src:.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-mem-boundary-capacity-final}"
mkdir -p "$MPLCONFIGDIR"

python -m mp_runners.mem_boundary_eval \
  --manifest_csv "$MANIFEST_CSV" \
  --source_root outputs/job_sets/mem_gen_boundary_capacity_20260427 \
  --out_dir "$OUT_DIR" \
  --expected_manifest_count 97 \
  --allow_checkpoint_count_mismatch \
  --manifest_index "$SLURM_ARRAY_TASK_ID" \
  --sample_count 10000 \
  --fid_reference_count 10000 \
  --heldout_split valid \
  --checkpoint_steps 1000000 \
  --device cuda \
  --write_manifest \
  --skip_combined \
  "$@"
