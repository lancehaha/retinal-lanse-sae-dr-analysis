#!/bin/bash
#SBATCH --job-name=retinavlm-octave-anatomy-cache
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_octave_2025/octave-cache-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_octave_2025"
CODE="$HOME/retinavlm-scratch/jlens_map_7finding_v3_code"
export PYTHONPATH="$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
"$HOME/retinavlm-scratch/venv/bin/python" "$CODE/prepare_octave_anatomy_cache.py" \
  --archive "$ROOT/OCTAVE.zip" \
  --acquisition-receipt "$ROOT/archive_audit_complete.json" \
  --inventory "$ROOT/anatomy_inventory.json" \
  --split "$ROOT/octave_anatomy_exploratory_volume_split.json" \
  --output-root "$ROOT/anatomy_cache_v2"
