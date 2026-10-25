#!/bin/bash
#SBATCH --job-name=retinavlm-b5-gate
#SBATCH --partition=gpu
#SBATCH --constraint=xgpg
#SBATCH --gpus=a100-40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/b5-gate-%j.out

set -euo pipefail
RUN_ROOT="$HOME/retinavlm-scratch"
test "$(uname -m)" = x86_64
echo '28690d3ced5cdd3349100448b8f471cb56bb1690b0502c28148d055a266068fe  '"$HOME/official_six_biomarker_split_v1.csv" | sha256sum -c -
test -d "$RUN_ROOT/retinavlm_irf_validation/images"
mkdir -p "$RUN_ROOT/tmp" "$RUN_ROOT/f1_results"
export TMPDIR="$RUN_ROOT/tmp"
export HF_HOME="$RUN_ROOT/hf"
"$RUN_ROOT/venv/bin/python" "$HOME/evaluate_retinavlm_b5_released_gate.py" \
  --split-csv "$HOME/official_six_biomarker_split_v1.csv" \
  --validation-images "$RUN_ROOT/retinavlm_irf_validation/images" \
  --model-dir "$RUN_ROOT/checkpoint/RetinaVLM-Specialist" \
  --authors-code "$RUN_ROOT/SpecialistVLMs" \
  --code-dir "$HOME/retinavlm_smoke" \
  --out "$RUN_ROOT/f1_results/b5_released_validation_gate.json"
