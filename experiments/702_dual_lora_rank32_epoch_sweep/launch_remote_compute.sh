#!/usr/bin/env bash
set -euo pipefail

ROOT=/remote_compute/home/repos/quality
RUN=/remote_compute/home/runs/exp702
SOURCE=/remote_compute/home/runs/exp701
PY=/remote_compute/home/.venv-exp699/bin/python
PARENT="$ROOT/research/qwen3vl_lora_holdout.py"
WRAPPER="$ROOT/experiments/701_dual_lora_epoch_sweep/run_epoch.py"
DATA=/remote_compute/home/data/exp699/solution140_full_support_sources/data.csv
OOF=/remote_compute/home/data/exp699/eval/four_head_oof.npz
MANIFEST="$SOURCE/shared/manifest.tsv.gz"
VENDOR=/remote_compute/home/data/exp699/vendor

run_arm() {
  local gpu=$1 architecture=$2 epochs=$3 model=$4 model_class=$5 cache=$6 dependency=$7
  while test ! -f "$dependency/output/lora_holdout_report.json"; do
    if ! kill -0 "$(cat "$SOURCE/launcher.pid")" 2>/dev/null; then
      echo "rank16 dependency missing after launcher terminal: $dependency" >&2
      return 2
    fi
    sleep 15
  done
  local name="${architecture}_r32_e${epochs}" out="$RUN/$name"
  test ! -e "$out"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" EXP_ID=702 \
  ECUP_MODEL_ROOT="$model" ECUP_DATA="$DATA" ECUP_OOF="$OOF" \
  ECUP_MANIFEST="$MANIFEST" ECUP_OUTPUT_DIR="$out/output" ECUP_VENDOR="$VENDOR" \
  HOLDOUT_FOLD=0 TRAINING_MODE=hard MODEL_CLASS="$model_class" \
  USE_CHAT_BATCH="$([ "$model_class" = multimodal ] && echo 1 || echo 0)" \
  EXP701_IMAGE_VIEW="$out/images" PYTORCH_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false "$PY" -u "$WRAPPER" --parent "$PARENT" \
    --architecture "$architecture" --epochs "$epochs" --cache "$cache" \
    --lora-r 32 --lora-alpha 64 >"$out/run.log" 2>&1
}

mkdir -p "$RUN"
for epoch in 1 2 3 4 5; do
  run_arm "$((epoch-1))" qwen35_4b "$epoch" /models/qwen35_4b multimodal \
    "$SOURCE/shared/q35_fold0_cache" "$SOURCE/qwen35_4b_e$epoch" &
  echo "$! qwen35_4b $epoch" >>"$RUN/owned_pids.txt"
done
run_arm 5 qwen3vl_2b 1 /models/qwen3vl_2b image_text "$SOURCE/shared/q3_fold0_cache" "$SOURCE/qwen3vl_2b_e4" &
echo "$! qwen3vl_2b 1" >>"$RUN/owned_pids.txt"
run_arm 6 qwen3vl_2b 2 /models/qwen3vl_2b image_text "$SOURCE/shared/q3_fold0_cache" "$SOURCE/qwen3vl_2b_e5" &
echo "$! qwen3vl_2b 2" >>"$RUN/owned_pids.txt"
run_arm 7 qwen3vl_2b 3 /models/qwen3vl_2b image_text "$SOURCE/shared/q3_fold0_cache" "$SOURCE/qwen3vl_2b_e3" &
echo "$! qwen3vl_2b 3" >>"$RUN/owned_pids.txt"
wait "$(awk '$2=="qwen3vl_2b" && $3==1 {print $1}' "$RUN/owned_pids.txt")"
run_arm 5 qwen3vl_2b 4 /models/qwen3vl_2b image_text "$SOURCE/shared/q3_fold0_cache" "$SOURCE/qwen3vl_2b_e4" &
echo "$! qwen3vl_2b 4" >>"$RUN/owned_pids.txt"
wait "$(awk '$2=="qwen3vl_2b" && $3==2 {print $1}' "$RUN/owned_pids.txt")"
run_arm 6 qwen3vl_2b 5 /models/qwen3vl_2b image_text "$SOURCE/shared/q3_fold0_cache" "$SOURCE/qwen3vl_2b_e5" &
echo "$! qwen3vl_2b 5" >>"$RUN/owned_pids.txt"
wait

