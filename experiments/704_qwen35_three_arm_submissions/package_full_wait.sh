#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/704_qwen35_three_arm_submissions
PACK=$ROOT/experiments/716_distilled_140_adapter_replacement
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
SOURCE=$PACK/.local/allowed-140-base.zip
FREEZE=$PACK/.local/base_freeze_report.json
AUDIT=$ROOT/experiments/717_incumbent_component_evidence_audit/results/incumbent_evidence_audit.json
OUT=$BASE/results/full

for mode in gold_control hardneg_candidate rank_candidate; do
  while [[ ! -s "$BASE/results/full_refits/$mode/output_contract.json" ]]; do sleep 60; done
done
while [[ ! -s "$SOURCE" || ! -s "$FREEZE" || ! -s "$AUDIT" ]]; do sleep 60; done

mkdir -p "$BASE/.local" "$OUT"
SOURCE_SHA=$(sha256sum "$SOURCE" | cut -d" " -f1)
for mode in gold_control hardneg_candidate rank_candidate; do
  if [[ ! -s "$OUT/${mode}-package.json" ]]; then
    "$PYTHON" "$PACK/build_submission.py" \
      --base-submission "$SOURCE" --expected-base-sha256 "$SOURCE_SHA" \
      --base-freeze-report "$FREEZE" \
      --adapter-dir "$BASE/results/full_refits/$mode/adapter" \
      --full-refit-contract "$BASE/results/full_refits/$mode/output_contract.json" \
      --incumbent-evidence-audit "$AUDIT" --allow-control-mode \
      --output "$OUT/${mode}-submit.zip" \
      --report "$OUT/${mode}-package.json" \
      > "$BASE/.local/full-package-${mode}.log" 2>&1
  fi
done
if [[ ! -s "$OUT/manifest.json" ]]; then
  "$PYTHON" "$BASE/build_manifest.py" \
    --root "$OUT" --tier full_data --output "$OUT/manifest.json" \
    > "$BASE/.local/full-manifest.log" 2>&1
fi
