#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/705_qwen35_deadline_fast_distill
FULL=$ROOT/experiments/715_qwen35_distilled_full_refit
TEACHER=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
RUNTIME=$FULL/.local/runtime-full
OUTER_TARGETS=$BASE/.local/teacher-targets-full-f3
TARGETS=$BASE/.local/targets-full-f3

mkdir -p "$BASE/.local" "$BASE/results/full_refits"

run_arm() {
  local mode="$1" device="$2"
  local output="$BASE/results/full_refits/$mode"
  if [[ -s "$output/output_contract.json" ]]; then return; fi
  local teacher_args=()
  if [[ "$mode" != gold_control ]]; then
    teacher_args=(--teacher-target-dir "$TARGETS")
  fi
  env CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 \
    "$PYTHON" "$FULL/run_full.py" \
      --runtime-dir "$RUNTIME" "${teacher_args[@]}" \
      --mode "$mode" --deadline-fast-track --output-dir "$output" \
      > "$BASE/.local/full-refit-${mode}.log" 2>&1
}

# Gold needs no teacher scores and starts as soon as f0 releases GPU0.
(
  while [[ ! -s "$TEACHER/.local/output-f0/output_contract.json" ]]; do sleep 30; done
  run_arm gold_control 0
) & p0=$!

while [[ ! -s "$OUTER_TARGETS/teacher_target_contract.json" ]]; do
  sleep 30
done
if [[ ! -s "$TARGETS/teacher_target_contract.json" ]]; then
  "$PYTHON" "$BASE/build_f3_full_targets.py" \
    --runtime-dir "$RUNTIME" \
    --outer-target-dir "$OUTER_TARGETS" \
    --fold-prediction-dir "$TEACHER/.local/output-f3" \
    --output-dir "$TARGETS" > "$BASE/.local/build-targets.log" 2>&1
fi
# Candidate arms use physical GPUs 1 and 2, which remain occupied by the
# model-parallel f0 validation until its immutable output contract is published.
while [[ ! -s "$TEACHER/.local/output-f0/output_contract.json" ]]; do sleep 30; done
run_arm hardneg_candidate 1 & p1=$!
run_arm rank_candidate 2 & p2=$!
status=0
for pid in "$p0" "$p1" "$p2"; do wait "$pid" || status=1; done
exit "$status"
