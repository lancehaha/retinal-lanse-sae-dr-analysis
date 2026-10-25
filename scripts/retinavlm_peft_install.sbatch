#!/bin/bash
#SBATCH --job-name=retinavlm-peft
#SBATCH --partition=normal
#SBATCH --constraint=xcnd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/peft-install-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
mkdir -p "$ROOT/tmp" "$ROOT/uv-cache"
export TMPDIR="$ROOT/tmp" UV_CACHE_DIR="$ROOT/uv-cache"
"$HOME/.local/bin/uv" pip install --python "$ROOT/venv/bin/python" --no-deps 'peft==0.10.0'
"$ROOT/venv/bin/python" -c 'import importlib.metadata; print("peft", importlib.metadata.version("peft"))'
