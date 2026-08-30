#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/704_qwen35_three_arm_submissions
FULL=$ROOT/experiments/715_qwen35_distilled_full_refit
TEACHER=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
STUDENT=$ROOT/experiments/698_qwen35_teacher_guided_controls
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
DATA=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/data.csv
IMAGES=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/images/images
FOLDS=$ROOT/validation/grouped_text_v1/folds.csv

while [[ ! -s "$TEACHER/results/teacher_oof/teacher_oof_report.json" \
      || ! -s "$STUDENT/results/evaluation.json" \
      || ! -s "$STUDENT/results/connected_guard.json" ]]; do
  sleep 60
done

# The normal winner refit prepares these same immutable inputs first. If its
# promotion gate exits without preparing them, this auxiliary lane does so.
while [[ ! -s "$FULL/.local/targets-full/teacher_target_contract.json" ]] \
      && tmux has-session -t mlitvinov_q38_full_refit 2>/dev/null; do
  sleep 60
done
mkdir -p "$FULL/.local" "$BASE/.local" "$BASE/results/full_refits"
if [[ ! -s "$FULL/.local/runtime-full/runtime_audit.json" ]]; then
  "$PYTHON" "$FULL/build_full_runtime.py" \
    --data "$DATA" --folds "$FOLDS" --images "$IMAGES" \
    --output-dir "$FULL/.local/runtime-full" \
    > "$BASE/.local/build-full-runtime.log" 2>&1
fi
if [[ ! -s "$FULL/.local/targets-full/teacher_target_contract.json" ]]; then
  "$PYTHON" "$FULL/build_oof_targets.py" \
    --runtime-dir "$FULL/.local/runtime-full" \
    --teacher-oof "$TEACHER/results/teacher_oof/teacher_oof.csv" \
    --teacher-report "$TEACHER/results/teacher_oof/teacher_oof_report.json" \
    --output-dir "$FULL/.local/targets-full" \
    > "$BASE/.local/build-full-targets.log" 2>&1
fi

run_arm() {
  local mode="$1" device="$2"
  local output="$BASE/results/full_refits/$mode"
  if [[ -s "$output/output_contract.json" ]]; then return; fi
  env CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 \
    "$PYTHON" "$FULL/run_full.py" \
      --runtime-dir "$FULL/.local/runtime-full" \
      --teacher-target-dir "$FULL/.local/targets-full" \
      --evaluation "$STUDENT/results/evaluation.json" \
      --connected-guard "$STUDENT/results/connected_guard.json" \
      --mode "$mode" --output-dir "$output" \
      > "$BASE/.local/full-refit-${mode}.log" 2>&1
}

run_arm gold_control 1 & p0=$!
run_arm hardneg_candidate 2 & p1=$!
run_arm rank_candidate 3 & p2=$!
status=0
for pid in "$p0" "$p1" "$p2"; do wait "$pid" || status=1; done
exit "$status"
