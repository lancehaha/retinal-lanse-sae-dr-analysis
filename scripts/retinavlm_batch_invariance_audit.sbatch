#!/bin/bash
#SBATCH --job-name=retinavlm-batch-audit
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/batch-invariance-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
test "$(uname -m)" = x86_64
mkdir -p "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
"$ROOT/venv/bin/python" "$ROOT/jlens_map_code/audit_retinavlm_batch_invariance.py" \
  --map-dir "$ROOT/jlens_maps/B5" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --validation-images "$ROOT/retinavlm_irf_validation/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt" \
  --output-dir "$ROOT/jlens_maps/B5/batch_invariance"
