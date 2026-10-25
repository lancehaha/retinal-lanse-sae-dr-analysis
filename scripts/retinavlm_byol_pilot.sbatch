#!/bin/bash
#SBATCH --job-name=retinavlm-byol-pilot
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph5
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/byol-pilot-%j.out

set -euo pipefail

RUN_ROOT="$HOME/retinavlm-scratch"
ARCHIVE="$HOME/retinavlm_byol_train.tar.gz"
EXPECTED_SHA256="dec85282b00af3a101b75b0ef6fc87d7dd669f4a392e58a72a5a6574778fc485"

test "$(uname -m)" = "x86_64"
test -f "$ARCHIVE"
ACTUAL_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
test "$ACTUAL_SHA256" = "$EXPECTED_SHA256"
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"
tar -xzf "$ARCHIVE" -C "$RUN_ROOT"

"$RUN_ROOT/venv/bin/python" "$HOME/train_retinavlm_encoder_byol_pilot.py" \
  --data-dir "$RUN_ROOT/retinavlm_byol_train" \
  --checkpoint-dir "$RUN_ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$RUN_ROOT/SpecialistVLMs" \
  --out-checkpoint "$RUN_ROOT/checkpoints/encoder_byol_pilot.pt" \
  --out-metrics "$RUN_ROOT/checkpoints/encoder_byol_pilot_metrics.json" \
  --steps 200 --batch-size 32 --workers 4 --log-every 20
