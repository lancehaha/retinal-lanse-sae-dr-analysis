#!/bin/bash
#SBATCH --job-name=retinavlm-abc-aux-check
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/abc-aux-preflight-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/abc_training_code"
MAP="${MAP_SET:-$ROOT/jlens_maps/validated_map_set.json}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-$ROOT/abc_preflight}"
test "$(uname -m)" = x86_64
test -f "$ROOT/abc_preflight/one_step.json"
if [ ! -f "$MAP" ]; then
  echo "Auxiliary preflight skipped: no validated map set"
  exit 0
fi
"$ROOT/venv/bin/python" -c 'import json,sys; a=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); assert a["status"]=="matched_lora_one_step_preflight_complete" and m["status"]=="validated_retinavlm_olives_map_set"' "$ROOT/abc_preflight/one_step.json" "$MAP"
mkdir -p "$PREFLIGHT_ROOT" "$ROOT/tmp"
if [ ! -f "$PREFLIGHT_ROOT/one_step.json" ]; then
  cp "$ROOT/abc_preflight/one_step.json" "$PREFLIGHT_ROOT/one_step.json"
fi
cmp -s "$ROOT/abc_preflight/one_step.json" "$PREFLIGHT_ROOT/one_step.json"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
for ARM in B C; do
  "$ROOT/venv/bin/python" "$CODE/train_retinavlm_olives_abc.py" \
    --arm "$ARM" --seed 42 --total-steps 1 --stop-step 1 --save-every 1 \
    --rank 8 --alpha 16 --lr 1e-4 --report-weight 1.0 --auxiliary-weight 0.1 \
    --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
    --train-images "$ROOT/retinavlm_byol_train/images" \
    --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
    --authors-code "$ROOT/SpecialistVLMs" \
    --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
    --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
    --map-set "$MAP" \
    --out-dir "$PREFLIGHT_ROOT/$ARM"
done
