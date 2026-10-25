#!/bin/bash
#SBATCH --job-name=retinavlm-abc-state-edit
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/abc-intervention-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/abc_training_code"
: "${ARM:?Pass ARM=A, B, or C}"
: "${CHECKPOINT:?Pass the validation-selected checkpoint}"
: "${VALIDATION_DIR:?Pass the complete validation directory for this checkpoint}"
: "${SELECTION_LOCK:?Pass the frozen validation selection lock}"
test "$(uname -m)" = x86_64
test -f "$CHECKPOINT"
test -f "$SELECTION_LOCK"
test -d "$VALIDATION_DIR"
mkdir -p "$ROOT/tmp" "$ROOT/abc_posttraining_interventions"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
"$ROOT/venv/bin/python" "$CODE/evaluate_retinavlm_abc_posttraining_intervention.py" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --validation-dir "$VALIDATION_DIR" \
  --image-root "$ROOT/retinavlm_irf_validation/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
  --map-set "$ROOT/jlens_maps_singleton/validated_map_set_combined.json" \
  --checkpoint "$CHECKPOINT" \
  --selection-lock "$SELECTION_LOCK" \
  --output "$ROOT/abc_posttraining_interventions/$ARM.json" \
  --arm "$ARM" --seed 42 --per-category 4 --controls 4
