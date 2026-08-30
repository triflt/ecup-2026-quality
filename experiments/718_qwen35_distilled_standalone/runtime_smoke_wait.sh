#!/usr/bin/env bash
set -euo pipefail

BASE=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution/experiments/718_qwen35_distilled_standalone
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
DATA=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/data.csv
IMAGES=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/images/images
QWEN35=/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3.5-4B

while [[ ! -s "$BASE/results/selection.json" ]]; do
  sleep 60
done
if ! "$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("authorized") is True else 3)' "$BASE/results/selection.json"; then
  echo "Standalone 4B was not authorized; runtime smoke intentionally not run."
  exit 0
fi
while [[ ! -s "$BASE/results/package_report.json" ]]; do
  sleep 60
done

# Serialize against the fixed-replacement exp716 runtime on physical GPU0.
LOCK_DIR=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution/.local
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/mlitvinov_q38_runtime_gpu0.lock"
flock 9

COMMON=(
  --submission "$BASE/results/distilled-qwen35-standalone-submit.zip"
  --package-report "$BASE/results/package_report.json"
  --data "$DATA" --images "$IMAGES" --python "$PYTHON"
  --qwen35-model "$QWEN35"
)
if [[ ! -s "$BASE/results/runtime_preflight.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$BASE/runtime_preflight.py" \
    --report "$BASE/results/runtime_preflight.json" \
    > "$BASE/.local/runtime-preflight.log" 2>&1
fi
if [[ ! -s "$BASE/results/runtime_smoke_3.json" ]]; then
  "$PYTHON" "$BASE/runtime_smoke.py" "${COMMON[@]}" \
    --rows 3 --work-dir "$BASE/.local/runtime-smoke-3" \
    --report "$BASE/results/runtime_smoke_3.json" \
    > "$BASE/.local/runtime-smoke-3.log" 2>&1
fi
if [[ ! -s "$BASE/results/runtime_smoke_600.json" ]]; then
  "$PYTHON" "$BASE/runtime_smoke.py" "${COMMON[@]}" \
    --rows 600 --work-dir "$BASE/.local/runtime-smoke-600" \
    --report "$BASE/results/runtime_smoke_600.json" \
    > "$BASE/.local/runtime-smoke-600.log" 2>&1
fi
if [[ ! -s "$BASE/results/runtime_acceptance.json" ]]; then
  "$PYTHON" "$BASE/finalize_runtime.py" \
    --submission "$BASE/results/distilled-qwen35-standalone-submit.zip" \
    --package-report "$BASE/results/package_report.json" \
    --smoke-3 "$BASE/results/runtime_smoke_3.json" \
    --smoke-600 "$BASE/results/runtime_smoke_600.json" \
    --output "$BASE/results/runtime_acceptance.json" \
    > "$BASE/.local/runtime-acceptance.log" 2>&1
fi
