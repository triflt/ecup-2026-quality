#!/usr/bin/env bash
set -euo pipefail

BASE=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python

for fold in 0 1 2 3 4; do
  while [[ ! -s "$BASE/.local/output-f${fold}/output_contract.json" ]]; do
    sleep 60
  done
done

"$PYTHON" "$BASE/assemble_oof.py" \
  --experiment-dir "$BASE" \
  --output-dir "$BASE/results/teacher_oof" \
  > "$BASE/.local/assemble-oof.log" 2>&1
