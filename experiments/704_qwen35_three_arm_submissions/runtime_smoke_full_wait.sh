#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
BASE=$ROOT/experiments/704_qwen35_three_arm_submissions
PACK=$ROOT/experiments/716_distilled_140_adapter_replacement
PYTHON=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
DATA=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/data.csv
IMAGES=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/images/images
EMBED=/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3-VL-Embedding-2B
INSTRUCT=/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3-VL-2B-Instruct
QWEN35=/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3.5-4B
OUT=$BASE/results/full

while [[ ! -s "$OUT/manifest.json" ]]; do sleep 60; done
while tmux has-session -t mlitvinov_q38_full_refit 2>/dev/null \
   || tmux has-session -t mlitvinov_q38_three_full 2>/dev/null; do
  sleep 60
done

# Share the same physical-GPU0 lock as experiments 716 and 718. These smokes
# are deliberately delayed until training exits; they cannot perturb metrics.
LOCK_DIR=$ROOT/.local
mkdir -p "$LOCK_DIR" "$BASE/.local"
exec 9>"$LOCK_DIR/mlitvinov_q38_runtime_gpu0.lock"
flock 9

for mode in gold_control hardneg_candidate rank_candidate; do
  if [[ ! -s "$OUT/${mode}-runtime-smoke-3.json" ]]; then
    "$PYTHON" "$PACK/runtime_smoke.py" \
      --submission "$OUT/${mode}-submit.zip" \
      --package-report "$OUT/${mode}-package.json" \
      --data "$DATA" --images "$IMAGES" --python "$PYTHON" \
      --qwen-embed-model "$EMBED" \
      --qwen-instruct-model "$INSTRUCT" \
      --qwen35-model "$QWEN35" \
      --rows 3 --work-dir "$BASE/.local/runtime-smoke-${mode}-3" \
      --report "$OUT/${mode}-runtime-smoke-3.json" \
      > "$BASE/.local/runtime-smoke-${mode}-3.log" 2>&1
  fi
done
