#!/bin/bash
#SBATCH --job-name=retinavlm-abc-eval
#SBATCH --partition=gpu
#SBATCH --constraint=xgph
#SBATCH --gpus=a100-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/abc-eval-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
CODE="$ROOT/abc_training_code"
: "${ARM:?Pass ARM=A, B, or C through sbatch --export}"
: "${SPLIT:?Pass SPLIT=validation or test through sbatch --export}"
: "${START:?Pass START through sbatch --export}"
: "${STOP:?Pass STOP through sbatch --export}"
: "${SEED:?Pass SEED through sbatch --export}"
: "${CHECKPOINT:?Pass the exact immutable checkpoint path through sbatch --export}"
: "${MAP_SET:?Pass the validated singleton map-set path through sbatch --export}"
: "${EVAL_ROOT:?Pass an output root through sbatch --export}"
test "$(uname -m)" = x86_64
case "$ARM" in A|B|C) ;; *) echo "Invalid ARM: $ARM" >&2; exit 2 ;; esac
case "$SPLIT" in
  validation) IMAGE_ROOT="$ROOT/retinavlm_irf_validation/images" ;;
  test)
    IMAGE_ROOT="$ROOT/retinavlm_olives_f1_test/images"
    : "${SELECTION_LOCK:?Test requires a frozen validation selection lock}"
    test -f "$SELECTION_LOCK"
    ;;
  *) echo "Invalid SPLIT: $SPLIT" >&2; exit 2 ;;
esac
test -f "$CHECKPOINT"
test -f "$MAP_SET"
mkdir -p "$EVAL_ROOT/$ARM" "$ROOT/tmp"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp"
args=(
  --split-csv "$HOME/official_six_biomarker_split_v1.csv"
  --image-root "$IMAGE_ROOT"
  --model-dir "$ROOT/checkpoint/RetinaVLM-Specialist"
  --authors-code "$ROOT/SpecialistVLMs"
  --adapted-checkpoint "$ROOT/checkpoints/encoder_byol_full.pt"
  --adapter-checkpoint "$ROOT/adapter_alignment/adapted_byol/adapter_final.pt"
  --map-set "$MAP_SET"
  --checkpoint "$CHECKPOINT"
  --out-dir "$EVAL_ROOT/$ARM"
  --arm "$ARM" --split "$SPLIT" --seed "$SEED"
  --start "$START" --stop "$STOP" --report-tokens 160
)
if [ "$SPLIT" = test ]; then args+=(--selection-lock "$SELECTION_LOCK"); fi
"$ROOT/venv/bin/python" "$CODE/evaluate_retinavlm_olives_abc_segment.py" "${args[@]}"
