#!/bin/bash
#SBATCH --job-name=jlens-v3-fd-diagnostic
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/jlens-v3-fd-diagnostic-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/jlens_map_7finding_v3_code"
OUT="$ROOT/jlens_maps_7finding_v3_diagnostics/finite_difference_epsilon_ladder.json"
test ! -e "$OUT"
mkdir -p "$(dirname "$OUT")" "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
"$ROOT/venv/bin/python" "$CODE/diagnose_retinavlm_map_v3_finite_difference.py" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --train-images "$ROOT/retinavlm_byol_train/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
  --plan-root "$ROOT/jlens_maps_7finding_v3_train_gate_plan" \
  --map-root "$ROOT/jlens_maps_7finding_v3_alt_layers" \
  --output "$OUT"
