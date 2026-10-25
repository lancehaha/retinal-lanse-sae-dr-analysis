#!/bin/bash
#SBATCH --job-name=retinavlm-adapter-preflight
#SBATCH --partition=gpu
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/adapter-preflight-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
test "$(uname -m)" = x86_64
mkdir -p "$ROOT/tmp" "$ROOT/adapter_alignment/preflight"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
"$ROOT/venv/bin/python" "$ROOT/adapter_alignment_code/train_retinavlm_adapter_olives.py" \
  --variant adapted_byol \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --train-images "$ROOT/retinavlm_byol_train/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --out-dir "$ROOT/adapter_alignment/preflight" \
  --steps 2 --save-every 1 --seed 42 --lr 1e-5
