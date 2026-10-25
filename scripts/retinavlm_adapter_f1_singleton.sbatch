#!/bin/bash
#SBATCH --job-name=retinavlm-f1-singleton
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/h/heshi/retinavlm-scratch/adapter-f1-singleton-%A_%a.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/adapter_alignment_code"
test "$(uname -m)" = x86_64
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then VARIANT=released; else VARIANT=adapted_byol; fi
OUT="$ROOT/adapter_alignment_singleton/$VARIANT/f1"
if [ -e "$OUT/f1_metrics.json" ]; then
  echo "Singleton evaluation already has a final metrics file: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT" "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
"$ROOT/venv/bin/python" "$CODE/evaluate_retinavlm_adapter_olives.py" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --drss-join "$HOME/official_drss_join_v1.csv" \
  --train-images "$ROOT/retinavlm_byol_train/images" \
  --validation-images "$ROOT/retinavlm_irf_validation/images" \
  --test-images "$ROOT/retinavlm_olives_f1_test/images" \
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$ROOT/SpecialistVLMs" \
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt" \
  --adapter-checkpoint "$ROOT/adapter_alignment/$VARIANT/adapter_final.pt" \
  --out-dir "$OUT" \
  --batch-size 1
