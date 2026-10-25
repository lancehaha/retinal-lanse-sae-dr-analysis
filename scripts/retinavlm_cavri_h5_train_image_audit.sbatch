#!/bin/bash
#SBATCH --job-name=cavri-h5-images
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_cavri_h5_2026/cavri-train-images-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_cavri_h5_2026"
export PYTHONPATH="$HOME/retinavlm-scratch/external_octave_2025/vendor${PYTHONPATH:+:$PYTHONPATH}"
"$HOME/retinavlm-scratch/venv/bin/python" "$ROOT/audit_cavri_h5_train_images.py" \
  --source-root "$ROOT/source" \
  --output "$ROOT/train_image_geometry.json"
