#!/bin/bash
#SBATCH -A berzelius-2026-78
#SBATCH -J mp_mgb_all_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/evals/logs/%x_%j.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/evals/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par"
cd "$REPO_ROOT"
mkdir -p jobs/evals/logs

module load Mambaforge
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
mamba activate mem_param_env

export PYTHONPATH=src:.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-mem-boundary-all-smoke}"
mkdir -p "$MPLCONFIGDIR"

python -m mp_runners.mem_boundary_eval \
  --all_done \
  --source_root outputs/job_sets/mem_gen_boundary_20260427 \
  --out_dir outputs/evals/mem_boundary_20260427_all_done_quality_20260501_smoke \
  --manifest_indices 0,72 \
  --sample_count 10000 \
  --fid_reference_count 10000 \
  --heldout_split valid \
  --checkpoint_steps 0,1000000 \
  --device cuda \
  --write_manifest \
  "$@"
