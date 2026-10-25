#!/bin/bash
#SBATCH --job-name=retinavlm-octave-anatomy-inventory
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_octave_2025/octave-anatomy-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_octave_2025"
CODE="$HOME/retinavlm-scratch/jlens_map_7finding_v3_code"
export PYTHONPATH="$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
"$HOME/retinavlm-scratch/venv/bin/python" "$CODE/audit_octave_archive_anatomy.py" \
  --archive "$ROOT/OCTAVE.zip" \
  --acquisition-receipt "$ROOT/archive_audit_complete.json" \
  --output "$ROOT/anatomy_inventory.json"
