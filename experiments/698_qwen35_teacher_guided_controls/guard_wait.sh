#!/usr/bin/env bash
set -euo pipefail

BASE=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution/experiments/698_qwen35_teacher_guided_controls
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
GUARD=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution/validation/connected_family_guard_v2/rows.csv

while [[ ! -s "$BASE/results/evaluation.json" || ! -s "$BASE/results/evaluation.npz" ]]; do
  sleep 60
done

if [[ ! -s "$BASE/results/connected_guard.json" ]]; then
  "$PYTHON" "$BASE/connected_guard.py" \
    --evaluation "$BASE/results/evaluation.json" \
    --oof "$BASE/results/evaluation.npz" \
    --guard "$GUARD" \
    --output "$BASE/results/connected_guard.json" \
    > "$BASE/.local/connected-guard.log" 2>&1
fi
