#!/usr/bin/env bash
set -euo pipefail

BASE=${EXP697_BASE:-/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher}
PYTHON=${EXP697_PYTHON:-/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python}

wait_for_contract() {
  local fold="$1"
  local producer_session="$2"
  local contract="$BASE/.local/output-f${fold}/output_contract.json"
  while [[ ! -s "$contract" ]]; do
    if ! tmux has-session -t "$producer_session" 2>/dev/null; then
      echo "producer $producer_session exited without $contract" >&2
      exit 1
    fi
    sleep 60
  done
}

run_fold() {
  local fold="$1"
  local devices="$2"
  local contract="$BASE/.local/output-f${fold}/output_contract.json"
  if [[ -s "$contract" ]]; then
    echo "fold $fold already has a completed contract; skipping"
    return 0
  fi
  env CUDA_VISIBLE_DEVICES="$devices" PYTHONUNBUFFERED=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" "$BASE/run_fold.py" \
    --fold "$fold" \
    --runtime-dir "$BASE/.local/runtime-f${fold}" \
    --output-dir "$BASE/.local/output-f${fold}" \
    --micro-batch 2 \
    --grad-accum 8 \
    > "$BASE/.local/fold${fold}.log" 2>&1
}

case "${1:-}" in
  lane-a)
    wait_for_contract 0 mlitvinov_q38_f0
    run_fold 1 0,1,2,3
    run_fold 4 0,1,2,3
    ;;
  lane-b)
    wait_for_contract 3 mlitvinov_q38_f3
    run_fold 2 4,5,6,7
    ;;
  evaluate)
    for fold in 0 1 2 3 4; do
      while [[ ! -s "$BASE/.local/output-f${fold}/output_contract.json" ]]; do
        sleep 60
      done
    done
    mkdir -p "$BASE/results"
    "$PYTHON" "$BASE/evaluate.py" \
      "$BASE/.local/output-f0/predictions.jsonl" \
      "$BASE/.local/output-f1/predictions.jsonl" \
      "$BASE/.local/output-f2/predictions.jsonl" \
      "$BASE/.local/output-f3/predictions.jsonl" \
      "$BASE/.local/output-f4/predictions.jsonl" \
      --out "$BASE/results/metrics.json" \
      > "$BASE/.local/evaluate.log" 2>&1
    "$PYTHON" "$BASE/assemble_oof.py" \
      --experiment-dir "$BASE" \
      --output-dir "$BASE/results/teacher_oof" \
      > "$BASE/.local/assemble-oof.log" 2>&1
    ;;
  *)
    echo "usage: $0 {lane-a|lane-b|evaluate}" >&2
    exit 2
    ;;
esac
