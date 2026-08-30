#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
TEACHER=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
BASE=$ROOT/experiments/706_qwen35_parent_anchored_distillation
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
TRAIN=$BASE/results/holdout_train_safe
EVAL=$BASE/results/holdout_eval
VALIDATION=$TEACHER/.local/runtime-f4/validation.jsonl

wait_file() {
  while [[ ! -s "$1" ]]; do sleep 20; done
}

wait_session_exit() {
  while tmux has-session -t "$1" 2>/dev/null; do sleep 20; done
}

score_update() {
  local update="$1"
  local device="$2"
  local dependency="${3:-}"
  local label
  printf -v label "%04d" "$update"
  local checkpoint=$TRAIN/checkpoints/update_$label
  local output=$EVAL/update_$label
  wait_file "$checkpoint/output_contract.json"
  if [[ -n "$dependency" ]]; then wait_session_exit "$dependency"; fi
  if [[ -s "$output/metrics.json" ]]; then return 0; fi
  env CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 \
    "$PYTHON" "$BASE/score_adapter.py" \
      --runtime-jsonl "$VALIDATION" \
      --adapter "$checkpoint/adapter" \
      --output-dir "$output" \
      > "$BASE/.local/eval-update-$label.log" 2>&1
}

mkdir -p "$EVAL" "$BASE/.local"
score_update 10 2 & p10=$!
score_update 20 3 & p20=$!
score_update 40 1 mlitvinov_q38_eval_parent & p40=$!
score_update 80 0 mlitvinov_q38_anchored_holdout & p80=$!
status=0
for pid in "$p10" "$p20" "$p40" "$p80"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "one or more checkpoint evaluations failed" >&2
  exit 1
fi
wait_file "$EVAL/parent/metrics.json"
if [[ ! -s "$BASE/results/holdout_selection.json" ]]; then
  "$PYTHON" "$BASE/select_checkpoint.py" \
    --eval-root "$EVAL" \
    --output "$BASE/results/holdout_selection.json" \
    > "$BASE/.local/select-holdout.log" 2>&1
fi
