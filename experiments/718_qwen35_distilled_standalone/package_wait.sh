#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/718_qwen35_distilled_standalone
SELECTION=$BASE/results/selection.json
FULL=$ROOT/experiments/715_qwen35_distilled_full_refit/results/full_refit
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python

while [[ ! -s "$SELECTION" ]]; do sleep 60; done
if ! "$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("authorized") is True else 3)' "$SELECTION"; then
  echo "Standalone 4B was not authorized; package intentionally not built."
  exit 0
fi
while [[ ! -s "$FULL/output_contract.json" ]]; do sleep 60; done
mkdir -p "$BASE/results" "$BASE/.local"
if [[ ! -s "$BASE/results/package_report.json" ]]; then
  "$PYTHON" "$BASE/build_package.py" \
    --selection "$SELECTION" \
    --full-refit-contract "$FULL/output_contract.json" \
    --adapter-dir "$FULL/adapter" \
    --run-source "$BASE/standalone_run.py" \
    --output "$BASE/results/distilled-qwen35-standalone-submit.zip" \
    --report "$BASE/results/package_report.json" \
    > "$BASE/.local/package.log" 2>&1
fi
