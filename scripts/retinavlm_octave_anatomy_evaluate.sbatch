#!/bin/bash
#SBATCH --job-name=octave-anatomy-eval
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_octave_2025/octave-eval-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_octave_2025"
"$HOME/retinavlm-scratch/venv/bin/python" "$ROOT/evaluate_octave_anatomy_feasibility.py" \
  --cache-root "$ROOT/anatomy_cache_v2" \
  --split "$ROOT/octave_anatomy_exploratory_volume_split.json" \
  --true-checkpoint "$ROOT/fit_true/epoch_20.pt" \
  --shifted-checkpoint "$ROOT/fit_shifted/epoch_20.pt" \
  --output-root "$ROOT/evaluation_paired"
