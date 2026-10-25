#!/bin/bash
#SBATCH --job-name=retinavlm-olives-f1
#SBATCH --partition=gpu
#SBATCH --constraint=xgpd
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/olives-f1-%j.out

set -euo pipefail
RUN_ROOT="$HOME/retinavlm-scratch"
TEST_ARCHIVE="$HOME/retinavlm_olives_f1_test.tar.gz"
test "$(uname -m)" = x86_64
test -f "$RUN_ROOT/retinavlm_byol_train/receipt.json"
test -f "$RUN_ROOT/retinavlm_irf_validation/receipt.json"
test -f "$RUN_ROOT/checkpoints/encoder_byol_full.pt"
echo 'f9d4d24112a735dbc486ebeffde17792f575924ae740dd2fa5e2aa7367cb898c  '"$TEST_ARCHIVE" | sha256sum -c -
echo '28690d3ced5cdd3349100448b8f471cb56bb1690b0502c28148d055a266068fe  '"$HOME/official_six_biomarker_split_v1.csv" | sha256sum -c -
echo '68b5fd90074f6acd50f609683fc08977b0ac53d85547b906d8cda42150a66229  '"$HOME/official_drss_join_v1.csv" | sha256sum -c -
mkdir -p "$RUN_ROOT/tmp" "$RUN_ROOT/f1_results"
export TMPDIR="$RUN_ROOT/tmp"
tar -xzf "$TEST_ARCHIVE" -C "$RUN_ROOT"

"$RUN_ROOT/venv/bin/python" "$HOME/evaluate_retinavlm_olives_f1.py" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --drss-join "$HOME/official_drss_join_v1.csv" \
  --train-images "$RUN_ROOT/retinavlm_byol_train/images" \
  --validation-images "$RUN_ROOT/retinavlm_irf_validation/images" \
  --test-images "$RUN_ROOT/retinavlm_olives_f1_test/images" \
  --released-checkpoint "$RUN_ROOT/checkpoint/RetinaVLM-Specialist" \
  --adapted-checkpoint "$RUN_ROOT/checkpoints/encoder_byol_full.pt" \
  --authors-code "$RUN_ROOT/SpecialistVLMs" \
  --out-dir "$RUN_ROOT/f1_results" \
  --batch-size 32 --workers 4
