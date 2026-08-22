#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <extracted-output-root>" >&2
  exit 2
fi

artifact_root=$1
result_root=experiments/310_qwen35_crossfit_soft_targets/results
predictions=()
for fold in 0 1 2 3 4; do
  prediction="$artifact_root/fold_${fold}/lora_holdout_predictions.csv"
  report="$artifact_root/fold_${fold}/lora_holdout_report.json"
  adapter="$artifact_root/fold_${fold}/adapter/adapter_model.safetensors"
  [[ -s "$prediction" && -s "$report" && -s "$adapter" ]] || {
    echo "incomplete fold $fold under $artifact_root" >&2
    exit 1
  }
  predictions+=("$prediction")
done
[[ -s "$artifact_root/full/adapter/adapter_model.safetensors" ]] || {
  echo "missing full adapter" >&2
  exit 1
}
[[ -s "$artifact_root/full/full_train_report.json" ]] || {
  echo "missing full train report" >&2
  exit 1
}

uv run --no-sync python research/aggregate_lora_oof.py \
  --predictions "${predictions[@]}" \
  --base-oof research/four-head-r2-extracted/four_head_oof.npz \
  --output "$result_root/soft_lora_oof_report.json"

PYTHONPATH=research uv run --no-sync python research/qwen35_locked_190_audit.py \
  --candidate-rank-npz exp310 "$result_root/soft_lora_oof_report.npz" lora_rank \
  --folds validation/grouped_text_v1/folds.csv \
  --output "$result_root/acceptance_audit.json"

PYTHONPATH=research uv run --no-sync python research/audit_connected_candidate.py \
  --predictions "$result_root/acceptance_audit.npz" \
  --candidate-key exp310_nested_predictions \
  --name exp310_soft_targets \
  --output "$result_root/connected_safe_audit.json"

PYTHONPATH=research uv run --no-sync python research/audit_component_decision_survival.py \
  --data research/data.csv \
  --predictions "$result_root/acceptance_audit.npz" \
  --candidate exp310 \
  --output "$result_root/prior_replay.json"

PYTHONPATH=research uv run --no-sync python research/audit_sports_error_transfer.py \
  --data research/data.csv \
  --predictions "$result_root/acceptance_audit.npz" \
  --candidate-key exp310_nested_predictions \
  --output "$result_root/sports_error_audit.json"

uv run --no-sync python research/finalize_qwen_soft_target.py \
  --locked "$result_root/acceptance_audit.json" \
  --connected "$result_root/connected_safe_audit.json" \
  --priors "$result_root/prior_replay.json" \
  --sports "$result_root/sports_error_audit.json" \
  --output "$result_root/post310_decision.json"
