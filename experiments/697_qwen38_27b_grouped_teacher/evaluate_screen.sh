#!/usr/bin/env bash
set -euo pipefail

BASE=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python

for fold in 0 3; do
  while [[ ! -s "$BASE/.local/output-f${fold}/output_contract.json" ]]; do
    sleep 60
  done
done

mkdir -p "$BASE/results"
"$PYTHON" "$BASE/evaluate.py" \
  "$BASE/.local/output-f0/predictions.jsonl" \
  "$BASE/.local/output-f3/predictions.jsonl" \
  --out "$BASE/results/screen_f0_f3_metrics.json" \
  > "$BASE/.local/evaluate-screen.log" 2>&1
