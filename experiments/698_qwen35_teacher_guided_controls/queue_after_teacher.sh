#!/usr/bin/env bash
set -euo pipefail

TEACHER=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
BASE=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution/experiments/698_qwen35_teacher_guided_controls
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
DATA=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/data.csv
FOLDS=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution/validation/grouped_text_v1/folds.csv

wait_for_file() {
  local path="$1"
  while [[ ! -s "$path" ]]; do
    sleep 60
  done
}

wait_for_session_exit() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
}

wait_all() {
  local status=0
  local pid
  for pid in "$@"; do
    wait "$pid" || status=1
  done
  return "$status"
}

run_target() {
  local fold="$1"
  local devices="$2"
  local output="$TEACHER/.local/teacher-targets-f${fold}"
  if [[ -s "$output/teacher_target_contract.json" ]]; then
    return
  fi
  env CUDA_VISIBLE_DEVICES="$devices" PYTHONUNBUFFERED=1 \
    "$PYTHON" "$TEACHER/generate_teacher_targets.py" \
    --fold "$fold" \
    --runtime-dir "$TEACHER/.local/runtime-f${fold}" \
    --adapter-dir "$TEACHER/.local/output-f${fold}/adapter" \
    --output-dir "$output" \
    > "$TEACHER/.local/teacher-targets-f${fold}.log" 2>&1
}

run_student_fold() {
  local mode="$1"
  local fold="$2"
  local device="$3"
  local output="$BASE/.local/${mode}-f${fold}"
  if [[ -s "$output/output_contract.json" ]]; then
    return
  fi
  local teacher_args=()
  if [[ "$mode" != gold_control ]]; then
    teacher_args=(--teacher-target-dir "$TEACHER/.local/teacher-targets-f${fold}")
  fi
  env CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 \
    "$PYTHON" "$BASE/run_fold.py" \
    --fold "$fold" \
    --mode "$mode" \
    --runtime-dir "$TEACHER/.local/runtime-f${fold}" \
    "${teacher_args[@]}" \
    --output-dir "$output" \
    > "$BASE/.local/${mode}-f${fold}.log" 2>&1
}

case "${1:-}" in
  targets-a)
    for fold in 0 1 4; do
      wait_for_file "$TEACHER/.local/output-f${fold}/output_contract.json"
    done
    while tmux has-session -t mlitvinov_q38_queue_a 2>/dev/null; do sleep 60; done
    for fold in 0 1 4; do run_target "$fold" 0,1,2,3; done
    ;;
  targets-b)
    for fold in 3 2; do
      wait_for_file "$TEACHER/.local/output-f${fold}/output_contract.json"
    done
    wait_for_session_exit mlitvinov_q38_queue_b
    # Gold does not consume teacher targets.  Run it first on the early-free
    # B lane, then give all four devices to the 27B target generator.
    wait_for_session_exit mlitvinov_q38_gold
    for fold in 3 2; do run_target "$fold" 4,5,6,7; done
    ;;
  gold-early)
    wait_for_file "$TEACHER/.local/output-f2/output_contract.json"
    wait_for_session_exit mlitvinov_q38_queue_b
    mkdir -p "$BASE/.local" "$BASE/results"
    (run_student_fold gold_control 0 4; run_student_fold gold_control 4 4) & p0=$!
    run_student_fold gold_control 1 5 & p1=$!
    run_student_fold gold_control 2 6 & p2=$!
    run_student_fold gold_control 3 7 & p3=$!
    wait_all "$p0" "$p1" "$p2" "$p3"
    ;;
  students-b)
    for fold in 3 2; do
      wait_for_file "$TEACHER/.local/teacher-targets-f${fold}/teacher_target_contract.json"
    done
    wait_for_session_exit mlitvinov_q38_targets_b
    mkdir -p "$BASE/.local" "$BASE/results"
    run_student_fold hardneg_candidate 3 4 & p0=$!
    run_student_fold hardneg_candidate 2 5 & p1=$!
    run_student_fold rank_candidate 3 6 & p2=$!
    run_student_fold rank_candidate 2 7 & p3=$!
    wait_all "$p0" "$p1" "$p2" "$p3"
    ;;
  students-a)
    for fold in 0 1 4; do
      wait_for_file "$TEACHER/.local/teacher-targets-f${fold}/teacher_target_contract.json"
    done
    wait_for_session_exit mlitvinov_q38_targets_a
    mkdir -p "$BASE/.local" "$BASE/results"
    (run_student_fold hardneg_candidate 0 0; run_student_fold rank_candidate 0 0) & p0=$!
    (run_student_fold hardneg_candidate 1 1; run_student_fold rank_candidate 1 1) & p1=$!
    (run_student_fold hardneg_candidate 4 2; run_student_fold rank_candidate 4 2) & p2=$!
    wait_all "$p0" "$p1" "$p2"
    ;;
  students-evaluate)
    for mode in gold_control hardneg_candidate rank_candidate; do
      for fold in 0 1 2 3 4; do
        wait_for_file "$BASE/.local/${mode}-f${fold}/output_contract.json"
      done
    done
    mkdir -p "$BASE/results"
    "$PYTHON" "$BASE/evaluate.py" \
      --data "$DATA" \
      --folds "$FOLDS" \
      --outputs "$BASE/.local" \
      --output "$BASE/results/evaluation.json" \
      > "$BASE/.local/evaluate.log" 2>&1
    ;;
  students)
    for fold in 0 1 2 3 4; do
      wait_for_file "$TEACHER/.local/teacher-targets-f${fold}/teacher_target_contract.json"
    done
    mkdir -p "$BASE/.local" "$BASE/results"
    for mode in gold_control hardneg_candidate rank_candidate; do
      smoke="$BASE/.local/smoke-${mode}"
      if [[ ! -s "$smoke/output_contract.json" ]]; then
        teacher_args=()
        if [[ "$mode" != gold_control ]]; then
          teacher_args=(--teacher-target-dir "$TEACHER/.local/teacher-targets-f0")
        fi
        env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
          "$PYTHON" "$BASE/run_fold.py" \
          --fold 0 --mode "$mode" --technical-smoke \
          --runtime-dir "$TEACHER/.local/runtime-f0" \
          "${teacher_args[@]}" \
          --output-dir "$smoke" \
          > "$BASE/.local/smoke-${mode}.log" 2>&1
      fi
    done
    for mode in gold_control hardneg_candidate rank_candidate; do
      pids=()
      for fold in 0 1 2 3 4; do
        run_student_fold "$mode" "$fold" "$fold" &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do wait "$pid"; done
    done
    "$PYTHON" "$BASE/evaluate.py" \
      --data "$DATA" \
      --folds "$FOLDS" \
      --outputs "$BASE/.local" \
      --output "$BASE/results/evaluation.json" \
      > "$BASE/.local/evaluate.log" 2>&1
    ;;
  *)
    echo "usage: $0 {targets-a|targets-b|gold-early|students-a|students-b|students-evaluate|students}" >&2
    exit 2
    ;;
esac
