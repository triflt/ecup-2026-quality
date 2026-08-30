#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/716_distilled_140_adapter_replacement
FULL=$ROOT/experiments/715_qwen35_distilled_full_refit/results/full_refit
AUDIT=$ROOT/experiments/717_incumbent_component_evidence_audit/results/incumbent_evidence_audit.json
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python

while [[ ! -s "$FULL/output_contract.json" ]]; do
  sleep 60
done
if [[ ! -s "$AUDIT" ]]; then
  echo "missing incumbent component evidence audit: $AUDIT" >&2
  exit 1
fi
mkdir -p "$BASE/results" "$BASE/.local"
SOURCE="$BASE/.local/allowed-140-base.zip"
FREEZE="$BASE/.local/base_freeze_report.json"
if [[ ! -s "$SOURCE" || ! -s "$FREEZE" ]]; then
  echo "missing provisioned no-OCR exp140 base or its freeze report" >&2
  exit 1
fi
SOURCE_SHA=$(sha256sum "$SOURCE" | cut -d" " -f1)
if [[ ! -s "$BASE/results/package_report.json" ]]; then
  "$PYTHON" "$BASE/build_submission.py" \
    --base-submission "$SOURCE" \
    --expected-base-sha256 "$SOURCE_SHA" \
    --base-freeze-report "$FREEZE" \
    --adapter-dir "$FULL/adapter" \
    --full-refit-contract "$FULL/output_contract.json" \
    --incumbent-evidence-audit "$AUDIT" \
    --output "$BASE/results/distilled-140-submit.zip" \
    --report "$BASE/results/package_report.json" \
    > "$BASE/.local/package.log" 2>&1
fi
