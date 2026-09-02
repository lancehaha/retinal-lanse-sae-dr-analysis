#!/bin/zsh

set -u

external_root="/Volumes/Elements/Retina-LanSE"
project_root="/Users/hgh54913/Documents/ChatGPT/CP4106 SAE ON RETINAL IMAGE ANALYSIS"
log_path="$external_root/experiments/full_train_extraction.terminal.log"
status_path="$external_root/experiments/full_train_extraction.exit_status.json"

echo "Retina-LanSE: resuming frozen RETFound train extraction"
echo "External output: $external_root/embeddings/global_final_mean_patch/train_cpu_fp32"
echo "Keep this Terminal tab and the Elements drive connected until completion."

caffeinate -i "$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/extract_retfound_embeddings.py" \
  --dataset-root "$external_root/datasets/eyepacs" \
  --manifest "$external_root/datasets/eyepacs/manifest.jsonl" \
  --checkpoint "$external_root/model_weights/retfound_cfp/RETFound_mae_natureCFP.pth" \
  --output "$external_root/embeddings/global_final_mean_patch/train_cpu_fp32" \
  --split train \
  --device cpu \
  --batch-size 4 \
  --workers 2 \
  --shard-size 1000 \
  --representation final_mean_patch 2>&1 | tee "$log_path"
run_status=${pipestatus[1]}

printf '{"exit_code": %d}\n' "$run_status" > "$status_path"
echo "Retina-LanSE extraction exit code: $run_status"
exit "$run_status"
