#!/bin/bash
# Auto-generated resume wrapper for non-DONE capacity-control tasks.
# Source manifest: jobs/job_sets/mem_gen_boundary_capacity_20260427/resume_undone_20260501.tsv

#SBATCH -A berzelius-2026-78
#SBATCH -J mp_mgb_cap_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --chdir=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/mem_gen_boundary_capacity_20260427/slurm_resume_20260501/%x_%A_%a.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/mem_gen_boundary_capacity_20260427/slurm_resume_20260501/%x_%A_%a.err
#SBATCH --array=15,16,17,18,19,20,21,24,25,26,27,28,29,30,31,32,33,34,35,50,52,53,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107%8

set -euo pipefail

REPO_ROOT="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par"
RESUME_MANIFEST="/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/mem_gen_boundary_capacity_20260427/resume_undone_20260501.tsv"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

RESUME_FROM_ATTEMPT="$(awk -F '	' -v tid="${TASK_ID}" 'NR > 1 && $1 == tid { print $9; found=1 } END { if (!found) exit 1 }' "${RESUME_MANIFEST}")"
if [[ -z "${RESUME_FROM_ATTEMPT}" || ! -d "${RESUME_FROM_ATTEMPT}" ]]; then
  echo "Could not resolve RESUME_FROM_ATTEMPT for task ${TASK_ID} from ${RESUME_MANIFEST}" >&2
  exit 2
fi

export MP_REPO_ROOT="${REPO_ROOT}"
export RESUME_FROM_ATTEMPT
export WANDB_MODE="${WANDB_MODE:-online}"

cd "${REPO_ROOT}"
exec bash jobs/job_sets/mem_gen_boundary_capacity_20260427/submit.sh
