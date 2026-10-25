#!/bin/bash
#SBATCH --job-name=cavri-h5-morphology
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_cavri_h5_2026/cavri-morphology-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_cavri_h5_2026"
export PYTHONPATH="$HOME/retinavlm-scratch/external_octave_2025/vendor${PYTHONPATH:+:$PYTHONPATH}"
"$HOME/retinavlm-scratch/venv/bin/python" "$ROOT/inspect_cavri_h5_train_morphology.py" \
  --archive "$ROOT/source/CAVRI-H5_CLAN_tiff.zip" \
  --geometry "$ROOT/train_image_geometry.json" \
  --output "$ROOT/VMA_VMT_train_morphology_sheet.png"
