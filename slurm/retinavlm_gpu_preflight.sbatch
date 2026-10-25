#!/usr/bin/env bash
#SBATCH --job-name=retinavlm-gpu-check
#SBATCH --partition=gpu
#SBATCH --gpus=a100-80
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=retinavlm-gpu-check-%j.out

set -euo pipefail

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "started=$(date -Is)"
echo "allocated_gpus=${SLURM_JOB_GPUS:-unknown}"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "finished=$(date -Is)"
