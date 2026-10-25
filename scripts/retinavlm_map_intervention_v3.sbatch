#!/bin/bash
#SBATCH --job-name=jlens-v3-map-edit
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/jlens-map-edit-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/jlens_map_7finding_v3_code"
MAP_ROOT="${MAP_ROOT:-$ROOT/jlens_maps_7finding_v3_train_gate}"
PLAN_ROOT="${PLAN_ROOT:-$ROOT/jlens_maps_7finding_v3_train_gate_plan}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-1}"
FINDING="${FINDING:?set FINDING explicitly}"
MAP="$MAP_ROOT/$FINDING"
test -n "$FINDING"
case "$FINDING" in B1|B2|B3|B4|B5|B6|SRF) ;; *) echo "Invalid finding: $FINDING" >&2; exit 2 ;; esac
test "$(uname -m)" = x86_64
STATUS="$("$ROOT/venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$MAP/statistical_gate.json")"
if [ "$STATUS" != candidate_statistical_gate_passed_needs_intervention ]; then
  "$ROOT/venv/bin/python" -c 'import json,sys; json.dump({"status":"state_intervention_skipped_statistical_gate_failed","statistical_gate_status":sys.argv[2]},open(sys.argv[1],"w"),indent=2)' "$MAP/state_intervention.json" "$STATUS"
  echo "State intervention skipped: statistical candidate gate failed"
  exit 0
fi
mkdir -p "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
export PYTHONPATH="$CODE:$ROOT/jlens_map_code${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/venv/bin/python" "$CODE/intervene_retinavlm_olives_map_v3.py" \
  --map-dir "$MAP" \
  --selection-plan "$PLAN_ROOT/${FINDING}_selection_plan.json" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --train-images "$ROOT/retinavlm_byol_train/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
  --output "$MAP/state_intervention.json" \
  --reference-batch-size "$REFERENCE_BATCH_SIZE"
