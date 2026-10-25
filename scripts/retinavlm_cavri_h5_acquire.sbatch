#!/bin/bash
#SBATCH --job-name=cavri-h5-audit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_cavri_h5_2026/cavri-acquire-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_cavri_h5_2026"
"$HOME/retinavlm-scratch/venv/bin/python" "$ROOT/acquire_cavri_h5_external.py" --output-root "$ROOT/source"
