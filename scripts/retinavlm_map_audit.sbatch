#!/bin/bash
#SBATCH --job-name=retinavlm-map-audit
#SBATCH --partition=normal
#SBATCH --constraint=xcnd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/jlens-map-audit-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/jlens_map_code"
MAP_ROOT="${MAP_ROOT:-$ROOT/jlens_maps}"
SELECTION="$MAP_ROOT/validation_pilot_selection.json"
FINDING="${FINDING:-auto}"
if [ "$FINDING" = remaining ]; then
  FINDING="$("$ROOT/venv/bin/python" "$CODE/resolve_retinavlm_map_finding.py" \
    --selection "$SELECTION" \
    --index "${SLURM_ARRAY_TASK_ID:?remaining finding requires array task index}")"
fi
if [ "$FINDING" = auto ]; then
  FINDING="$("$ROOT/venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen_finding"])' "$SELECTION")"
fi
test -n "$FINDING"
case "$FINDING" in B1|B2|B3|B4|B5|B6|SRF) ;; *) echo "Invalid finding: $FINDING" >&2; exit 2 ;; esac
"$ROOT/venv/bin/python" "$CODE/audit_retinavlm_olives_map_candidate.py" \
  --map-dir "$MAP_ROOT/$FINDING" \
  --output "$MAP_ROOT/$FINDING/statistical_gate.json"
