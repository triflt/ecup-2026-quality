#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/705_qwen35_deadline_fast_distill
FULL=$ROOT/experiments/715_qwen35_distilled_full_refit
TEACHER=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
RUNTIME=$BASE/.local/teacher-runtime-full-f3
OUTPUT=$BASE/.local/teacher-targets-full-f3

mkdir -p "$BASE/.local"
if [[ ! -s "$RUNTIME/runtime_audit.json" ]]; then
  "$PYTHON" "$BASE/build_teacher_runtime.py" \
    --full-runtime "$FULL/.local/runtime-full" --output-dir "$RUNTIME" \
    > "$BASE/.local/build-teacher-runtime.log" 2>&1
fi
if [[ ! -s "$OUTPUT/teacher_target_contract.json" ]]; then
  env CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 \
    "$PYTHON" "$TEACHER/generate_teacher_targets.py" \
      --fold 3 --runtime-dir "$RUNTIME" \
      --adapter-dir "$TEACHER/.local/output-f3/adapter" \
      --output-dir "$OUTPUT" \
      > "$BASE/.local/teacher-targets-full-f3.log" 2>&1
fi
