#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/718_qwen35_distilled_standalone
STUDENT=$ROOT/experiments/698_qwen35_teacher_guided_controls/results
REFIT=$ROOT/experiments/715_qwen35_distilled_full_refit/results/full_refit/output_contract.json
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python

while [[ ! -s "$STUDENT/evaluation.json" || ! -s "$STUDENT/connected_guard.json" ]]; do
  sleep 60
done
mkdir -p "$BASE/results" "$BASE/.local"
if "$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("promoted_candidates") else 3)' "$STUDENT/connected_guard.json"; then
  while [[ ! -s "$REFIT" ]]; do sleep 60; done
  refit_args=(--full-refit-contract "$REFIT")
else
  refit_args=()
fi
if [[ ! -s "$BASE/results/selection.json" ]]; then
  "$PYTHON" "$BASE/select_candidate.py" \
    --evaluation "$STUDENT/evaluation.json" \
    --connected-guard "$STUDENT/connected_guard.json" \
    "${refit_args[@]}" \
    --output "$BASE/results/selection.json" \
    > "$BASE/.local/selection.log" 2>&1
fi
