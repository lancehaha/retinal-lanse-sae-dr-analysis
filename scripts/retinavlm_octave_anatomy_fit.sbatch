#!/bin/bash
#SBATCH --job-name=octave-anatomy-fit
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_octave_2025/octave-fit-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_octave_2025"
ARM="${ARM:?Set ARM to true or shifted}"
OUT="${OUT:?Set isolated OUT directory}"
EXTRA=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then EXTRA+=(--dry-run); fi
"$HOME/retinavlm-scratch/venv/bin/python" "$ROOT/train_octave_anatomy_feasibility.py" \
  --cache-root "$ROOT/anatomy_cache_v2" \
  --split "$ROOT/octave_anatomy_exploratory_volume_split.json" \
  --schedule "$ROOT/anatomy_train_schedule.json" \
  --preflight "$ROOT/anatomy_preflight.json" \
  --protocol "$ROOT/RETINAVLM_OCTAVE_ANATOMY_EXPLORATORY_PROTOCOL_20261024.md" \
  --arm "$ARM" --output-root "$OUT" "${EXTRA[@]}"
