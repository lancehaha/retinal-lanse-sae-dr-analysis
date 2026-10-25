#!/bin/bash
#SBATCH --job-name=retinavlm-abc-train
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/abc-train-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/abc_training_code"
MAP="${MAP_SET:-$ROOT/jlens_maps/validated_map_set.json}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-$ROOT/abc_preflight}"
TRAINING_ROOT="${TRAINING_ROOT:-$ROOT/abc_training}"
: "${ARM:?Pass ARM=A, B, or C through sbatch --export}"
: "${STOP_STEP:?Pass STOP_STEP through sbatch --export}"
: "${TOTAL_STEPS:?Pass TOTAL_STEPS through sbatch --export}"
: "${SEED:?Pass SEED through sbatch --export}"
: "${RANK:?Pass RANK through sbatch --export}"
: "${ALPHA:?Pass ALPHA through sbatch --export}"
: "${LR:?Pass LR through sbatch --export}"
: "${REPORT_WEIGHT:?Pass REPORT_WEIGHT through sbatch --export}"
: "${AUXILIARY_WEIGHT:?Pass AUXILIARY_WEIGHT through sbatch --export}"
test "$(uname -m)" = x86_64
test -f "$MAP"
test -f "$PREFLIGHT_ROOT/one_step.json"
test -f "$PREFLIGHT_ROOT/B/segment_000001_receipt.json"
test -f "$PREFLIGHT_ROOT/C/segment_000001_receipt.json"
"$ROOT/venv/bin/python" "$CODE/verify_retinavlm_abc_preflight.py" \
  --preflight-root "$PREFLIGHT_ROOT" \
  --map-set "$MAP" \
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
  --report-weight "$REPORT_WEIGHT" --auxiliary-weight "$AUXILIARY_WEIGHT" \
  --seed "$SEED"
mkdir -p "$TRAINING_ROOT" "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
"$ROOT/venv/bin/python" "$CODE/train_retinavlm_olives_abc.py" \
  --arm "$ARM" --seed "$SEED" --total-steps "$TOTAL_STEPS" \
  --stop-step "$STOP_STEP" --save-every 500 \
  --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
  --report-weight "$REPORT_WEIGHT" --auxiliary-weight "$AUXILIARY_WEIGHT" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --train-images "$ROOT/retinavlm_byol_train/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
  --map-set "$MAP" \
  --out-dir "$TRAINING_ROOT/${ARM}_seed${SEED}"
