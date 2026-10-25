#!/bin/bash
#SBATCH --job-name=retinavlm-abc-map-controls
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/abc-map-controls-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/abc_training_code"
: "${ARM:?Pass ARM=A, B, or C}"
: "${CHECKPOINT:?Pass the validation-selected A/B/C checkpoint}"
: "${SELECTION_LOCK:?Pass the frozen validation selection lock}"
test "$(uname -m)" = x86_64
test -f "$CHECKPOINT"
test -f "$SELECTION_LOCK"
mkdir -p "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
"$ROOT/venv/bin/python" "$CODE/evaluate_retinavlm_abc_posttraining_map_controls.py" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --image-root "$ROOT/retinavlm_irf_validation/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
  --map-set "$ROOT/jlens_maps_singleton/validated_map_set_combined.json" \
  --checkpoint "$CHECKPOINT" \
  --selection-lock "$SELECTION_LOCK" \
  --out-dir "$ROOT/abc_posttraining_map_controls/$ARM" \
  --arm "$ARM" --seed 42 --max-patients 13
