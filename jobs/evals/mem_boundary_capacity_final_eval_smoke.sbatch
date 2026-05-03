#!/bin/bash
#SBATCH -A berzelius-2026-78
#SBATCH -J mp_mgb_cap_eval_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/evals/logs/%x_%j.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/evals/logs/%x_%j.err

set -euo pipefail

# This smoke job is intentionally isolated from the real eval outputs.
# It reads completed capacity training attempts and writes only to the smoke out_dir.
REPO_ROOT="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par"
cd "$REPO_ROOT"
mkdir -p jobs/evals/logs

module load Mambaforge
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
mamba activate mem_param_env

export PYTHONPATH=src:.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-mem-boundary-capacity-smoke}"
mkdir -p "$MPLCONFIGDIR"

# MANIFEST_INDICES can be overridden after inspecting the smoke manifest.
# Sample/reference counts are deliberately tiny because this checks plumbing, not paper numbers.
python -m mp_runners.mem_boundary_eval \
  --all_done \
  --source_root outputs/job_sets/mem_gen_boundary_capacity_20260427 \
  --out_dir "${OUT_DIR:-outputs/evals/mem_boundary_capacity_20260427_final_quality_20260503_smoke}" \
  --expected_manifest_count 0 \
  --allow_checkpoint_count_mismatch \
  --manifest_indices "${MANIFEST_INDICES:-0}" \
  --sample_count "${SAMPLE_COUNT:-256}" \
  --fid_reference_count "${FID_REFERENCE_COUNT:-256}" \
  --heldout_split valid \
  --checkpoint_steps 1000000 \
  --device cuda \
  --write_manifest \
  --skip_combined \
  "$@"
