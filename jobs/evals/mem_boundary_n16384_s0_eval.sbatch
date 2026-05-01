#!/bin/bash
#SBATCH -A berzelius-2026-78
#SBATCH -J mp_mgb_all_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --array=0-83%4
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/evals/logs/%x_%A_%a.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/evals/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par"
cd "$REPO_ROOT"
mkdir -p jobs/evals/logs

module load Mambaforge
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
mamba activate mem_param_env

export PYTHONPATH=src:.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-mem-boundary-all-eval}"
mkdir -p "$MPLCONFIGDIR"

python -m mp_runners.mem_boundary_eval \
  --all_done \
  --source_root outputs/job_sets/mem_gen_boundary_20260427 \
  --out_dir outputs/evals/mem_boundary_20260427_all_done_quality_20260501 \
  --manifest_index "$SLURM_ARRAY_TASK_ID" \
  --sample_count 10000 \
  --fid_reference_count 10000 \
  --heldout_split valid \
  --device cuda \
  --skip_combined \
  "$@"
