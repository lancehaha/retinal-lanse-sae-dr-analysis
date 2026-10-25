#!/bin/bash
#SBATCH --job-name=octave-v2-fit
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpe1
#SBATCH --constraint=xgpe
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_octave_2025/octave-v2-fit-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_octave_2025"
ARM="${ARM:?Set ARM to true or shifted}"
OUT="${OUT:?Set isolated OUT directory}"
EXTRA=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then EXTRA+=(--dry-run); fi
"$HOME/retinavlm-scratch/venv/bin/python" "$ROOT/train_octave_anatomy_v2.py" \
  --cache-root "$ROOT/anatomy_cache_v2" \
  --split "$ROOT/octave_anatomy_v2_exploratory_split.json" \
  --schedule "$ROOT/anatomy_v2_train_schedule.json" \
  --preflight "$ROOT/anatomy_v2_preflight.json" \
  --protocol "$ROOT/RETINAVLM_OCTAVE_ANATOMY_EXPLORATORY_V2_PROTOCOL_20261024.md" \
  --arm "$ARM" --output-root "$OUT" "${EXTRA[@]}"
