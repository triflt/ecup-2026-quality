#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/719_distillation_final_selection
EVAL=$ROOT/experiments/698_qwen35_teacher_guided_controls/results/evaluation.json
GUARD=$ROOT/experiments/698_qwen35_teacher_guided_controls/results/connected_guard.json
AUDIT=$ROOT/experiments/717_incumbent_component_evidence_audit/results/incumbent_evidence_audit.json
INCUMBENT_METRICS=$ROOT/experiments/140_dual_lora_fusion/results/metrics.json
STANDALONE_SELECTION=$ROOT/experiments/718_qwen35_distilled_standalone/results/selection.json
INCUMBENT_ZIP=$ROOT/experiments/716_distilled_140_adapter_replacement/.local/allowed-140-base.zip
INCUMBENT_PACKAGE=$ROOT/experiments/716_distilled_140_adapter_replacement/.local/base_freeze_report.json
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python

for path in "$EVAL" "$GUARD" "$AUDIT" "$INCUMBENT_METRICS" "$STANDALONE_SELECTION" "$INCUMBENT_ZIP" "$INCUMBENT_PACKAGE"; do
  while [[ ! -s "$path" ]]; do sleep 60; done
done
mkdir -p "$BASE/results" "$BASE/.local"
ARGS=(
  --evaluation "$EVAL" --guard "$GUARD" --incumbent-audit "$AUDIT" --incumbent-metrics "$INCUMBENT_METRICS"
  --standalone-selection "$STANDALONE_SELECTION"
  --incumbent-zip "$INCUMBENT_ZIP" --incumbent-package "$INCUMBENT_PACKAGE"
)
if "$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("promoted_candidates") else 3)' "$GUARD"; then
  FULL=$ROOT/experiments/715_qwen35_distilled_full_refit/results/full_refit/output_contract.json
  FIXED_ZIP=$ROOT/experiments/716_distilled_140_adapter_replacement/results/distilled-140-submit.zip
  FIXED_PACKAGE=$ROOT/experiments/716_distilled_140_adapter_replacement/results/package_report.json
  FIXED_RUNTIME=$ROOT/experiments/716_distilled_140_adapter_replacement/results/runtime_acceptance.json
  for path in "$FULL" "$FIXED_ZIP" "$FIXED_PACKAGE" "$FIXED_RUNTIME"; do
    while [[ ! -s "$path" ]]; do sleep 60; done
  done
  ARGS+=(--full-refit "$FULL" --fixed-zip "$FIXED_ZIP" --fixed-package "$FIXED_PACKAGE" --fixed-runtime "$FIXED_RUNTIME")
fi
if "$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("authorized") is True else 3)' "$STANDALONE_SELECTION"; then
  STANDALONE_ZIP=$ROOT/experiments/718_qwen35_distilled_standalone/results/distilled-qwen35-standalone-submit.zip
  STANDALONE_PACKAGE=$ROOT/experiments/718_qwen35_distilled_standalone/results/package_report.json
  STANDALONE_RUNTIME=$ROOT/experiments/718_qwen35_distilled_standalone/results/runtime_acceptance.json
  for path in "$STANDALONE_ZIP" "$STANDALONE_PACKAGE" "$STANDALONE_RUNTIME"; do
    while [[ ! -s "$path" ]]; do sleep 60; done
  done
  ARGS+=(--standalone-zip "$STANDALONE_ZIP" --standalone-package "$STANDALONE_PACKAGE" --standalone-runtime "$STANDALONE_RUNTIME")
fi
if [[ ! -s "$BASE/results/final_selection.json" ]]; then
  "$PYTHON" "$BASE/select_final.py" "${ARGS[@]}" \
    --output "$BASE/results/final_selection.json" \
    > "$BASE/.local/final-selection.log" 2>&1
fi
