#!/bin/bash
# Strict-fat FFHQ speed benchmark.
#
# Submit with:
#   sbatch jobs/job_sets/ffhq_speed_grid_20260426/submit_fat.sh

#SBATCH -A berzelius-2026-78
#SBATCH -J mp_ffhq_spd_f
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=fat
#SBATCH --time=04:00:00
#SBATCH --chdir=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par
#SBATCH --output=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/ffhq_speed_grid_20260426/slurm_fat/%x_%A_%a.out
#SBATCH --error=/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/ffhq_speed_grid_20260426/slurm_fat/%x_%A_%a.err
#SBATCH --array=0-23%24

set -euo pipefail

export NODE_CLASS="fat"
source "/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/jobs/job_sets/ffhq_speed_grid_20260426/run_one.sh"
