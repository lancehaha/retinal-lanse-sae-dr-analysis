#!/bin/bash
#SBATCH --job-name=jlens-v3-b6-state-diagnostic
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/jlens-v3-b6-state-diagnostic-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/jlens_map_7finding_v3_code"
MAP="$ROOT/jlens_maps_7finding_v3_train_gate/B6"
OUT="$ROOT/jlens_maps_7finding_v3_diagnostics/B6_layers24_30_exploratory_state_intervention.json"
test ! -e "$OUT"
test "$(uname -m)" = x86_64
mkdir -p "$(dirname "$OUT")" "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
export PYTHONPATH="$CODE:$ROOT/jlens_map_code${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/venv/bin/python" "$CODE/diagnose_retinavlm_b6_v3_state_influence.py" \
  --map-dir "$MAP" \
  --selection-plan "$ROOT/jlens_maps_7finding_v3_train_gate_plan/B6_selection_plan.json" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --train-images "$ROOT/retinavlm_byol_train/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
  --output "$OUT" \
  --per-category 4 --controls 4 --reference-batch-size 1
