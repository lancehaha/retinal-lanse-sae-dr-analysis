#!/bin/bash
#SBATCH --job-name=retinavlm-map-combine
#SBATCH --partition=normal
#SBATCH --constraint=xcnd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/jlens-map-combine-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch"
MAP_ROOT="$ROOT/jlens_maps_singleton"
"$ROOT/venv/bin/python" \
  "$ROOT/jlens_map_code/finalize_retinavlm_singleton_map_set.py" \
  --map-root "$MAP_ROOT" \
  --inventory "$MAP_ROOT/final_map_gate_inventory_${SLURM_JOB_ID}.json" \
  --output "$MAP_ROOT/validated_map_set_combined.json"
