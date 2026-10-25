#!/bin/bash
#SBATCH --job-name=retinavlm-irf-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph5
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/encoder-irf-probe-%j.out

set -euo pipefail
RUN_ROOT="$HOME/retinavlm-scratch"
ARCHIVE="$HOME/retinavlm_irf_validation.tar.gz"
EXPECTED_SHA256="c02384b37f505aad3f1ebe3db6017f364e56fec885799148ab05f9f3f435ea95"

test "$(uname -m)" = "x86_64"
test -f "$ARCHIVE"
test "$(sha256sum "$ARCHIVE" | awk '{print $1}')" = "$EXPECTED_SHA256"
test -f "$RUN_ROOT/retinavlm_byol_train/receipt.json"
test -f "$RUN_ROOT/checkpoints/encoder_byol_full.pt"
mkdir -p "$RUN_ROOT/tmp" "$RUN_ROOT/validation_results"
export TMPDIR="$RUN_ROOT/tmp"
tar -xzf "$ARCHIVE" -C "$RUN_ROOT"

"$RUN_ROOT/venv/bin/python" "$HOME/evaluate_retinavlm_encoder_irf_probe.py" \
  --train-dir "$RUN_ROOT/retinavlm_byol_train" \
  --validation-dir "$RUN_ROOT/retinavlm_irf_validation" \
  --released-checkpoint "$RUN_ROOT/checkpoint/RetinaVLM-Specialist" \
  --adapted-checkpoint "$RUN_ROOT/checkpoints/encoder_byol_full.pt" \
  --authors-code "$RUN_ROOT/SpecialistVLMs" \
  --out-metrics "$RUN_ROOT/validation_results/encoder_irf_probe_metrics.json" \
  --out-predictions "$RUN_ROOT/validation_results/encoder_irf_probe_predictions.csv" \
  --batch-size 64 --workers 4
