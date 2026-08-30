#!/usr/bin/env bash
set -euo pipefail

TEACHER=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
FAST=$ROOT/experiments/705_qwen35_deadline_fast_distill/.local/teacher-targets-full-f3
while [[ ! -s "$FAST/teacher_target_contract.json" ]]; do
  sleep 30
done
while tmux has-session -t mlitvinov_q38_fast_targets_full 2>/dev/null; do sleep 10; done
bash "$TEACHER/queue_remaining.sh" lane-b
