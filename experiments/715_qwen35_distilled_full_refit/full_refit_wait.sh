#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/715_qwen35_distilled_full_refit
TEACHER=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
STUDENT=$ROOT/experiments/698_qwen35_teacher_guided_controls
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
DATA=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/data.csv
IMAGES=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/images/images
FOLDS=$ROOT/validation/grouped_text_v1/folds.csv

while [[ ! -s "$TEACHER/results/teacher_oof/teacher_oof_report.json" \
      || ! -s "$STUDENT/results/connected_guard.json" ]]; do
  sleep 60
done

if ! "$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("promoted_candidates") else 3)' "$STUDENT/results/connected_guard.json"; then
  echo "No candidate passed both experiment-698 gates; full refit is not authorized."
  exit 0
fi

mkdir -p "$BASE/.local" "$BASE/results"
if [[ ! -s "$BASE/.local/runtime-full/runtime_audit.json" ]]; then
  "$PYTHON" "$BASE/build_full_runtime.py" \
    --data "$DATA" --folds "$FOLDS" --images "$IMAGES" \
    --output-dir "$BASE/.local/runtime-full" \
    > "$BASE/.local/build-full-runtime.log" 2>&1
fi
if [[ ! -s "$BASE/.local/targets-full/teacher_target_contract.json" ]]; then
  "$PYTHON" "$BASE/build_oof_targets.py" \
    --runtime-dir "$BASE/.local/runtime-full" \
    --teacher-oof "$TEACHER/results/teacher_oof/teacher_oof.csv" \
    --teacher-report "$TEACHER/results/teacher_oof/teacher_oof_report.json" \
    --output-dir "$BASE/.local/targets-full" \
    > "$BASE/.local/build-full-targets.log" 2>&1
fi

COMMON=(
  --runtime-dir "$BASE/.local/runtime-full"
  --teacher-target-dir "$BASE/.local/targets-full"
  --evaluation "$STUDENT/results/evaluation.json"
  --connected-guard "$STUDENT/results/connected_guard.json"
)
if [[ ! -s "$BASE/.local/smoke/output_contract.json" ]]; then
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$PYTHON" "$BASE/run_full.py" "${COMMON[@]}" \
    --technical-smoke --output-dir "$BASE/.local/smoke" \
    > "$BASE/.local/smoke.log" 2>&1
fi
if [[ ! -s "$BASE/results/full_refit/output_contract.json" ]]; then
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$PYTHON" "$BASE/run_full.py" "${COMMON[@]}" \
    --output-dir "$BASE/results/full_refit" \
    > "$BASE/.local/full-refit.log" 2>&1
fi
