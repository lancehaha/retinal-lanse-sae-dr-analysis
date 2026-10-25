#!/bin/bash
#SBATCH --job-name=retinavlm-jlens-map
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/jlens-map-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="${MAP_CODE:-$ROOT/jlens_map_code}"
FINDING="${FINDING:-auto}"
MAP_ROOT="${MAP_ROOT:-$ROOT/jlens_maps}"
F1_ROOT="${F1_ROOT:-$ROOT/adapter_alignment}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CONTEXTS_PER_CLASS="${CONTEXTS_PER_CLASS:-16}"
CALIBRATION_PER_CLASS="${CALIBRATION_PER_CLASS:-64}"
CONTROL_PER_CLASS="${CONTROL_PER_CLASS:-16}"
SELECTION_PLAN="${SELECTION_PLAN:-}"
MAP_LAYERS="${MAP_LAYERS:-8 16 24 30}"
test "$(uname -m)" = x86_64
test -f "$F1_ROOT/adapted_byol/f1/f1_metrics.json"
test -f "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt"
mkdir -p "$MAP_ROOT" "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
if [ "$FINDING" = remaining ]; then
  FINDING="$("$ROOT/venv/bin/python" "$CODE/resolve_retinavlm_map_finding.py" \
    --selection "$MAP_ROOT/validation_pilot_selection.json" \
    --index "${SLURM_ARRAY_TASK_ID:?remaining finding requires array task index}")"
fi
if [ "$FINDING" = auto ]; then
  FINDING="$("$ROOT/venv/bin/python" "$CODE/select_retinavlm_jlens_finding.py" \
    --validation-predictions "$F1_ROOT/adapted_byol/f1/validation_predictions.csv" \
    --output "$MAP_ROOT/validation_pilot_selection.json")"
fi
case "$FINDING" in B1|B2|B3|B4|B5|B6|SRF) ;; *) echo "Invalid finding: $FINDING" >&2; exit 2 ;; esac
mkdir -p "$MAP_ROOT/$FINDING"
selection_args=()
if [ -n "$SELECTION_PLAN" ]; then
  test -f "$SELECTION_PLAN"
  selection_args=(--selection-plan "$SELECTION_PLAN")
fi
read -r -a layer_values <<< "$MAP_LAYERS"
test "${#layer_values[@]}" -ge 2
"$ROOT/venv/bin/python" "$CODE/fit_retinavlm_olives_jlens_map.py" \
  --finding "$FINDING" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --train-images "$ROOT/retinavlm_byol_train/images" \
  --validation-images "$ROOT/retinavlm_irf_validation/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
  --out-dir "$MAP_ROOT/$FINDING" \
  --batch-size "$BATCH_SIZE" \
  --contexts-per-class "$CONTEXTS_PER_CLASS" \
  --calibration-per-class "$CALIBRATION_PER_CLASS" \
  --control-per-class "$CONTROL_PER_CLASS" \
  --layers "${layer_values[@]}" \
  "${selection_args[@]}" \
  ${PATIENT_DISJOINT_FIT:+--patient-disjoint-fit}
