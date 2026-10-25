#!/bin/bash
#SBATCH --job-name=retinavlm-adapter-align
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/h/heshi/retinavlm-scratch/adapter-align-%A_%a.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/adapter_alignment_code"
test "$(uname -m)" = x86_64
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then VARIANT=released; else VARIANT=adapted_byol; fi
mkdir -p "$ROOT/adapter_alignment/$VARIANT" "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
"$ROOT/venv/bin/python" "$CODE/train_retinavlm_adapter_olives.py" \
  --variant "$VARIANT" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --train-images "$ROOT/retinavlm_byol_train/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --out-dir "$ROOT/adapter_alignment/$VARIANT" \
  --steps 10000 --save-every 500 --seed 42 --lr 1e-5
