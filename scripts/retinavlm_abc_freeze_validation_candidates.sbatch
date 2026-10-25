#!/bin/bash
#SBATCH --job-name=retinavlm-abc-candidates
#SBATCH --partition=normal
#SBATCH --constraint=xcnd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/abc-validation-candidates-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
test "$(uname -m)" = x86_64
"$ROOT/venv/bin/python" \
  "$ROOT/abc_training_code/build_retinavlm_abc_validation_candidates.py" \
  --map-set "$ROOT/jlens_maps_singleton/validated_map_set_combined.json" \
  --training-root "$ROOT/abc_training" \
  --evaluation-root "$ROOT/abc_validation_singleton" \
  --output "$ROOT/abc_validation_singleton/prespecified_candidates.json" \
  --seed 42
