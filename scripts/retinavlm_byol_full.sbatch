#!/bin/bash
#SBATCH --job-name=retinavlm-byol-full
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph5
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/byol-full-%j.out

set -euo pipefail
RUN_ROOT="$HOME/retinavlm-scratch"
test "$(uname -m)" = "x86_64"
test -f "$RUN_ROOT/retinavlm_byol_train/receipt.json"
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"

"$RUN_ROOT/venv/bin/python" "$HOME/train_retinavlm_encoder_byol_full.py" \
  --data-dir "$RUN_ROOT/retinavlm_byol_train" \
  --checkpoint-dir "$RUN_ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$RUN_ROOT/SpecialistVLMs" \
  --out-checkpoint "$RUN_ROOT/checkpoints/encoder_byol_full.pt" \
  --out-metrics "$RUN_ROOT/checkpoints/encoder_byol_full_metrics.json" \
  --steps 2083 --batch-size 32 --workers 4 --log-every 100
