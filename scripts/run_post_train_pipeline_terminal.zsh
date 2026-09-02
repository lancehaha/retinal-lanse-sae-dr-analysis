#!/bin/zsh

set -u

external_root="/Volumes/Elements/Retina-LanSE"
project_root="/Users/hgh54913/Documents/ChatGPT/CP4106 SAE ON RETINAL IMAGE ANALYSIS"
manifest="$external_root/datasets/eyepacs/manifest_publication_v1.jsonl"
train_evidence="$external_root/embeddings/global_final_mean_patch/train_cpu_fp32/extraction_evidence.json"
validation_output="$external_root/embeddings/global_final_mean_patch/validation_cpu_fp32"
search_output="$external_root/experiments/paper_global_sae_screen_mean_patch"
log_path="$external_root/experiments/post_train_pipeline.terminal.log"
status_path="$external_root/experiments/post_train_pipeline.exit_status.json"

exec > >(tee -a "$log_path") 2>&1
caffeinate -i -w $$ &
caffeinate_pid=$!
trap 'kill "$caffeinate_pid" >/dev/null 2>&1 || true' EXIT
printf '{"stage":"pipeline","status":"running"}\n' > "$status_path"

echo "Retina-LanSE: waiting for completed train extraction evidence"
while [[ ! -f "$train_evidence" ]]; do
  sleep 30
done

echo "Train evidence detected; starting required near-duplicate leakage audit"
near_duplicate_audit="$external_root/datasets/eyepacs/near_duplicate_audit.json"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/audit_near_duplicate_leakage.py" \
  --dataset-root "$external_root/datasets/eyepacs" \
  --manifest "$manifest" \
  --output "$near_duplicate_audit" \
  --workers 2
near_duplicate_status=$?
if [[ "$near_duplicate_status" -ne 0 ]]; then
  printf '{"stage":"near_duplicate_audit","exit_code":%d}\n' "$near_duplicate_status" > "$status_path"
  exit "$near_duplicate_status"
fi

echo "Near-duplicate audit passed; starting separate validation extraction"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/extract_retfound_embeddings.py" \
  --dataset-root "$external_root/datasets/eyepacs" \
  --manifest "$manifest" \
  --checkpoint "$external_root/model_weights/retfound_cfp/RETFound_mae_natureCFP.pth" \
  --output "$validation_output" \
  --split validation \
  --device cpu \
  --batch-size 4 \
  --workers 2 \
  --shard-size 1000 \
  --representation final_mean_patch
validation_status=$?
if [[ "$validation_status" -ne 0 ]]; then
  printf '{"stage":"validation_extraction","exit_code":%d}\n' "$validation_status" > "$status_path"
  exit "$validation_status"
fi

echo "Validation extraction complete; starting the full 54-candidate label-free SAE screen"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/run_global_sae_search.py" \
  --config "$project_root/configs/paper_global_sae.yaml" \
  --train-run "$external_root/embeddings/global_final_mean_patch/train_cpu_fp32" \
  --validation-run "$validation_output" \
  --output "$search_output" \
  --device cpu
search_status=$?
if [[ "$search_status" -ne 0 ]]; then
  printf '{"stage":"global_sae_screen","exit_code":%d}\n' "$search_status" > "$status_path"
  exit "$search_status"
fi

echo "Global SAE screen complete; starting three-finalist, five-seed confirmation"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/run_global_sae_finalists.py" \
  --config "$project_root/configs/paper_global_sae.yaml" \
  --screen-output "$search_output" \
  --train-run "$external_root/embeddings/global_final_mean_patch/train_cpu_fp32" \
  --validation-run "$validation_output" \
  --output "$external_root/experiments/paper_global_sae_finalists_mean_patch" \
  --device cpu
finalist_status=$?
if [[ "$finalist_status" -ne 0 ]]; then
  printf '{"stage":"global_sae_finalists_mean_patch","exit_code":%d}\n' "$finalist_status" > "$status_path"
  exit "$finalist_status"
fi

echo "Mean-patch finalist selected; fitting frozen label-free validation baselines"
mean_baseline_output="$external_root/experiments/paper_representation_baselines_mean_patch"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/run_representation_baselines.py" \
  --train-run "$external_root/embeddings/global_final_mean_patch/train_cpu_fp32" \
  --target-run "$validation_output" \
  --selection "$external_root/experiments/paper_global_sae_finalists_mean_patch/final_selection.json" \
  --output "$mean_baseline_output" \
  --target-split validation
mean_baseline_status=$?
if [[ "$mean_baseline_status" -ne 0 ]]; then
  printf '{"stage":"representation_baselines_mean_patch","exit_code":%d}\n' "$mean_baseline_status" > "$status_path"
  exit "$mean_baseline_status"
fi

echo "Mean-patch selection frozen; starting the preregistered CLS pooling ablation"
cls_train_output="$external_root/embeddings/global_final_cls/train_cpu_fp32"
cls_validation_output="$external_root/embeddings/global_final_cls/validation_cpu_fp32"
cls_screen_output="$external_root/experiments/paper_global_sae_screen_cls"

"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/extract_retfound_embeddings.py" \
  --dataset-root "$external_root/datasets/eyepacs" \
  --manifest "$manifest" \
  --checkpoint "$external_root/model_weights/retfound_cfp/RETFound_mae_natureCFP.pth" \
  --output "$cls_train_output" \
  --split train \
  --device cpu \
  --batch-size 4 \
  --workers 2 \
  --shard-size 1000 \
  --representation final_cls
cls_train_status=$?
if [[ "$cls_train_status" -ne 0 ]]; then
  printf '{"stage":"cls_train_extraction","exit_code":%d}\n' "$cls_train_status" > "$status_path"
  exit "$cls_train_status"
fi

"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/extract_retfound_embeddings.py" \
  --dataset-root "$external_root/datasets/eyepacs" \
  --manifest "$manifest" \
  --checkpoint "$external_root/model_weights/retfound_cfp/RETFound_mae_natureCFP.pth" \
  --output "$cls_validation_output" \
  --split validation \
  --device cpu \
  --batch-size 4 \
  --workers 2 \
  --shard-size 1000 \
  --representation final_cls
cls_validation_status=$?
if [[ "$cls_validation_status" -ne 0 ]]; then
  printf '{"stage":"cls_validation_extraction","exit_code":%d}\n' "$cls_validation_status" > "$status_path"
  exit "$cls_validation_status"
fi

"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/run_global_sae_search.py" \
  --config "$project_root/configs/paper_global_sae.yaml" \
  --train-run "$cls_train_output" \
  --validation-run "$cls_validation_output" \
  --output "$cls_screen_output" \
  --device cpu
cls_screen_status=$?
if [[ "$cls_screen_status" -ne 0 ]]; then
  printf '{"stage":"global_sae_screen_cls","exit_code":%d}\n' "$cls_screen_status" > "$status_path"
  exit "$cls_screen_status"
fi

"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/run_global_sae_finalists.py" \
  --config "$project_root/configs/paper_global_sae.yaml" \
  --screen-output "$cls_screen_output" \
  --train-run "$cls_train_output" \
  --validation-run "$cls_validation_output" \
  --output "$external_root/experiments/paper_global_sae_finalists_cls" \
  --device cpu
cls_finalist_status=$?
if [[ "$cls_finalist_status" -ne 0 ]]; then
  printf '{"stage":"global_sae_finalists_cls","exit_code":%d}\n' "$cls_finalist_status" > "$status_path"
  exit "$cls_finalist_status"
fi

echo "CLS finalist selected; fitting frozen label-free validation baselines"
cls_baseline_output="$external_root/experiments/paper_representation_baselines_cls"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/run_representation_baselines.py" \
  --train-run "$cls_train_output" \
  --target-run "$cls_validation_output" \
  --selection "$external_root/experiments/paper_global_sae_finalists_cls/final_selection.json" \
  --output "$cls_baseline_output" \
  --target-split validation
cls_baseline_status=$?
if [[ "$cls_baseline_status" -ne 0 ]]; then
  printf '{"stage":"representation_baselines_cls","exit_code":%d}\n' "$cls_baseline_status" > "$status_path"
  exit "$cls_baseline_status"
fi

echo "Both representation selections frozen; creating locked-test authorization manifest"
runtime_snapshot="$external_root/experiments/experiment_1_runtime_snapshot.json"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/capture_runtime_snapshot.py" \
  --checkpoint "$external_root/model_weights/retfound_cfp/RETFound_mae_natureCFP.pth" \
  --dataset-manifest "$manifest" \
  --upstream-repo "$external_root/runtime/RETFound" \
  --output "$runtime_snapshot"
runtime_snapshot_status=$?
if [[ "$runtime_snapshot_status" -ne 0 ]]; then
  printf '{"stage":"runtime_snapshot","exit_code":%d}\n' "$runtime_snapshot_status" > "$status_path"
  exit "$runtime_snapshot_status"
fi

"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/freeze_internal_test_access.py" \
  --mean-patch-selection "$external_root/experiments/paper_global_sae_finalists_mean_patch/final_selection.json" \
  --cls-selection "$external_root/experiments/paper_global_sae_finalists_cls/final_selection.json" \
  --mean-patch-baselines "$mean_baseline_output/validation_baselines.json" \
  --cls-baselines "$cls_baseline_output/validation_baselines.json" \
  --dataset-manifest "$manifest" \
  --protocol "$project_root/research/EXPERIMENT_1_GLOBAL_SAE_PROTOCOL.md" \
  --config "$project_root/configs/paper_global_sae.yaml" \
  --runtime-snapshot "$runtime_snapshot" \
  --sae-repeatability-report "$external_root/experiments/software_check_sae_repeatability_report.json" \
  --near-duplicate-audit "$near_duplicate_audit" \
  --near-duplicate-remediation "$external_root/datasets/eyepacs/near_duplicate_remediation.json" \
  --output "$external_root/experiments/internal_test_freeze_manifest.json"
freeze_status=$?
if [[ "$freeze_status" -ne 0 ]]; then
  printf '{"stage":"internal_test_freeze","exit_code":%d}\n' "$freeze_status" > "$status_path"
  exit "$freeze_status"
fi

freeze_manifest="$external_root/experiments/internal_test_freeze_manifest.json"
mean_test_output="$external_root/embeddings/global_final_mean_patch/internal_test_cpu_fp32"
cls_test_output="$external_root/embeddings/global_final_cls/internal_test_cpu_fp32"
mean_test_baselines="$external_root/experiments/paper_representation_baselines_mean_patch_test"
cls_test_baselines="$external_root/experiments/paper_representation_baselines_cls_test"
mean_test_evaluation="$external_root/experiments/paper_locked_internal_test_mean_patch"
cls_test_evaluation="$external_root/experiments/paper_locked_internal_test_cls"

echo "Protocol frozen; opening the internal test split once for mean-patch extraction"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/extract_retfound_embeddings.py" \
  --dataset-root "$external_root/datasets/eyepacs" \
  --manifest "$manifest" \
  --checkpoint "$external_root/model_weights/retfound_cfp/RETFound_mae_natureCFP.pth" \
  --output "$mean_test_output" \
  --split internal_test \
  --device cpu \
  --batch-size 4 \
  --workers 2 \
  --shard-size 1000 \
  --representation final_mean_patch \
  --test-freeze-manifest "$freeze_manifest"
mean_test_extract_status=$?
if [[ "$mean_test_extract_status" -ne 0 ]]; then
  printf '{"stage":"mean_patch_internal_test_extraction","exit_code":%d}\n' "$mean_test_extract_status" > "$status_path"
  exit "$mean_test_extract_status"
fi

"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/run_representation_baselines.py" \
  --train-run "$external_root/embeddings/global_final_mean_patch/train_cpu_fp32" \
  --target-run "$mean_test_output" \
  --selection "$external_root/experiments/paper_global_sae_finalists_mean_patch/final_selection.json" \
  --output "$mean_test_baselines" \
  --target-split internal_test \
  --pca-artifact "$mean_baseline_output/pca_fit.pt"
mean_test_baseline_status=$?
if [[ "$mean_test_baseline_status" -ne 0 ]]; then
  printf '{"stage":"mean_patch_internal_test_baselines","exit_code":%d}\n' "$mean_test_baseline_status" > "$status_path"
  exit "$mean_test_baseline_status"
fi

if [[ ! -f "$mean_test_evaluation/locked_internal_test_evaluation.json" ]]; then
  "$external_root/runtime/venv/bin/python" \
    "$project_root/scripts/run_locked_internal_test_evaluation.py" \
    --freeze-manifest "$freeze_manifest" \
    --test-run "$mean_test_output" \
    --train-run "$external_root/embeddings/global_final_mean_patch/train_cpu_fp32" \
    --selection "$external_root/experiments/paper_global_sae_finalists_mean_patch/final_selection.json" \
    --baseline-report "$mean_test_baselines/internal_test_baselines.json" \
    --representation final_mean_patch \
    --output "$mean_test_evaluation" \
    --device cpu
  mean_test_evaluation_status=$?
  if [[ "$mean_test_evaluation_status" -ne 0 ]]; then
    printf '{"stage":"mean_patch_internal_test_evaluation","exit_code":%d}\n' "$mean_test_evaluation_status" > "$status_path"
    exit "$mean_test_evaluation_status"
  fi
else
  echo "Mean-patch locked-test report already exists; preserving the single-access result"
fi

echo "Opening the same frozen internal test split for the preregistered CLS ablation"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/extract_retfound_embeddings.py" \
  --dataset-root "$external_root/datasets/eyepacs" \
  --manifest "$manifest" \
  --checkpoint "$external_root/model_weights/retfound_cfp/RETFound_mae_natureCFP.pth" \
  --output "$cls_test_output" \
  --split internal_test \
  --device cpu \
  --batch-size 4 \
  --workers 2 \
  --shard-size 1000 \
  --representation final_cls \
  --test-freeze-manifest "$freeze_manifest"
cls_test_extract_status=$?
if [[ "$cls_test_extract_status" -ne 0 ]]; then
  printf '{"stage":"cls_internal_test_extraction","exit_code":%d}\n' "$cls_test_extract_status" > "$status_path"
  exit "$cls_test_extract_status"
fi

"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/run_representation_baselines.py" \
  --train-run "$cls_train_output" \
  --target-run "$cls_test_output" \
  --selection "$external_root/experiments/paper_global_sae_finalists_cls/final_selection.json" \
  --output "$cls_test_baselines" \
  --target-split internal_test \
  --pca-artifact "$cls_baseline_output/pca_fit.pt"
cls_test_baseline_status=$?
if [[ "$cls_test_baseline_status" -ne 0 ]]; then
  printf '{"stage":"cls_internal_test_baselines","exit_code":%d}\n' "$cls_test_baseline_status" > "$status_path"
  exit "$cls_test_baseline_status"
fi

if [[ ! -f "$cls_test_evaluation/locked_internal_test_evaluation.json" ]]; then
  "$external_root/runtime/venv/bin/python" \
    "$project_root/scripts/run_locked_internal_test_evaluation.py" \
    --freeze-manifest "$freeze_manifest" \
    --test-run "$cls_test_output" \
    --train-run "$cls_train_output" \
    --selection "$external_root/experiments/paper_global_sae_finalists_cls/final_selection.json" \
    --baseline-report "$cls_test_baselines/internal_test_baselines.json" \
    --representation final_cls \
    --output "$cls_test_evaluation" \
    --device cpu
  cls_test_evaluation_status=$?
  if [[ "$cls_test_evaluation_status" -ne 0 ]]; then
    printf '{"stage":"cls_internal_test_evaluation","exit_code":%d}\n' "$cls_test_evaluation_status" > "$status_path"
    exit "$cls_test_evaluation_status"
  fi
else
  echo "CLS locked-test report already exists; preserving the single-access result"
fi

echo "Locked-test reports complete; auditing every Experiment 1 evidence gate"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/audit_experiment_1.py" \
  --external-root "$external_root" \
  --output "$external_root/experiments/experiment_1_completion_audit.json"
audit_status=$?
if [[ "$audit_status" -ne 0 ]]; then
  printf '{"stage":"experiment_1_completion_audit","exit_code":%d}\n' "$audit_status" > "$status_path"
  exit "$audit_status"
fi

echo "Evidence audit passed; generating the paper-ready results report"
"$external_root/runtime/venv/bin/python" \
  "$project_root/scripts/generate_experiment_1_report.py" \
  --external-root "$external_root" \
  --audit "$external_root/experiments/experiment_1_completion_audit.json" \
  --output "$external_root/reports/EXPERIMENT_1_RESULTS.md"
report_status=$?
printf '{"stage":"experiment_1_results_report","exit_code":%d}\n' "$report_status" > "$status_path"
if [[ "$report_status" -eq 0 ]]; then
  echo "Retina-LanSE Experiment 1 completed, passed the evidence audit, and generated its results report"
fi
exit "$report_status"
