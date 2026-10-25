#!/bin/bash
#SBATCH --job-name=octave-v2-preflight
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_octave_2025/octave-v2-preflight-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_octave_2025"
PY="$HOME/retinavlm-scratch/venv/bin/python"
"$PY" "$ROOT/preflight_octave_anatomy_v2.py" \
  --cache-root "$ROOT/anatomy_cache_v2" \
  --split "$ROOT/octave_anatomy_v2_exploratory_split.json" \
  --protocol "$ROOT/RETINAVLM_OCTAVE_ANATOMY_EXPLORATORY_V2_PROTOCOL_20261024.md" \
  --output "$ROOT/anatomy_v2_preflight.json"
"$PY" "$ROOT/create_octave_anatomy_v2_schedule.py" \
  --cache-root "$ROOT/anatomy_cache_v2" \
  --split "$ROOT/octave_anatomy_v2_exploratory_split.json" \
  --protocol "$ROOT/RETINAVLM_OCTAVE_ANATOMY_EXPLORATORY_V2_PROTOCOL_20261024.md" \
  --preflight "$ROOT/anatomy_v2_preflight.json" \
  --output "$ROOT/anatomy_v2_train_schedule.json"
