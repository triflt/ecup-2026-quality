#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/705_qwen35_deadline_fast_distill
TEACHER=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
for mode in gold_control hardneg_candidate rank_candidate; do
  while [[ ! -s "$BASE/results/full_refits/$mode/output_contract.json" ]]; do sleep 30; done
done
while tmux has-session -t mlitvinov_q38_fast_full 2>/dev/null; do sleep 10; done
bash "$TEACHER/queue_remaining.sh" lane-a
