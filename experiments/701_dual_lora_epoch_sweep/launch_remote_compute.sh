#!/usr/bin/env bash
set -euo pipefail

ROOT=/remote_compute/home/repos/quality
RUN=/remote_compute/home/runs/exp701
PY=/remote_compute/home/.venv-exp699/bin/python
PARENT="$ROOT/research/qwen3vl_lora_holdout.py"
WRAPPER="$ROOT/experiments/701_dual_lora_epoch_sweep/run_epoch.py"
DATA=/remote_compute/home/data/exp699/solution140_full_support_sources/data.csv
OOF=/remote_compute/home/data/exp699/eval/four_head_oof.npz
MANIFEST="$RUN/shared/manifest.tsv.gz"
VENDOR=/remote_compute/home/data/exp699/vendor

while pgrep -f '[e]xp701/shared/q35_fold0_cache' >/dev/null; do sleep 5; done
test "$(find "$RUN/shared/q35_fold0_cache" -type f | wc -l | tr -d ' ')" = 7352
test "$(find "$RUN/shared/q3_fold0_cache" -type f | wc -l | tr -d ' ')" = 7352

run_arm() {
  local gpu=$1 architecture=$2 epochs=$3 model=$4 model_class=$5 cache=$6
  local name="${architecture}_e${epochs}"
  local out="$RUN/$name"
  test ! -e "$out"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" \
  ECUP_MODEL_ROOT="$model" \
  ECUP_DATA="$DATA" \
  ECUP_OOF="$OOF" \
  ECUP_MANIFEST="$MANIFEST" \
  ECUP_OUTPUT_DIR="$out/output" \
  ECUP_VENDOR="$VENDOR" \
  HOLDOUT_FOLD=0 TRAINING_MODE=hard MODEL_CLASS="$model_class" \
  USE_CHAT_BATCH="$([ "$model_class" = multimodal ] && echo 1 || echo 0)" \
  EXP701_IMAGE_VIEW="$out/images" \
  PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
  "$PY" -u "$WRAPPER" --parent "$PARENT" --architecture "$architecture" \
    --epochs "$epochs" --cache "$cache" >"$out/run.log" 2>&1
}

mkdir -p "$RUN"
for epoch in 1 2 3 4 5; do
  run_arm "$((epoch-1))" qwen35_4b "$epoch" /models/qwen35_4b multimodal "$RUN/shared/q35_fold0_cache" &
  echo "$! qwen35_4b $epoch" >>"$RUN/owned_pids.txt"
done
for epoch in 1 2 3; do
  run_arm "$((epoch+4))" qwen3vl_2b "$epoch" /models/qwen3vl_2b image_text "$RUN/shared/q3_fold0_cache" &
  echo "$! qwen3vl_2b $epoch" >>"$RUN/owned_pids.txt"
done

wait "$(awk '$2=="qwen3vl_2b" && $3==1 {print $1}' "$RUN/owned_pids.txt")"
run_arm 5 qwen3vl_2b 4 /models/qwen3vl_2b image_text "$RUN/shared/q3_fold0_cache" &
echo "$! qwen3vl_2b 4" >>"$RUN/owned_pids.txt"
wait "$(awk '$2=="qwen3vl_2b" && $3==2 {print $1}' "$RUN/owned_pids.txt")"
run_arm 6 qwen3vl_2b 5 /models/qwen3vl_2b image_text "$RUN/shared/q3_fold0_cache" &
echo "$! qwen3vl_2b 5" >>"$RUN/owned_pids.txt"
wait
