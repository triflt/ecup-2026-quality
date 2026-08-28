#!/usr/bin/env bash
set -euo pipefail

ROOT=/remote_compute/home/repos/quality
RUN=/remote_compute/home/runs/exp703_retry1
SOURCE=/remote_compute/home/runs/exp701
PY=/remote_compute/home/.venv-exp699/bin/python
PARENT="$ROOT/research/qwen3vl_lora_holdout.py"
WRAPPER="$ROOT/experiments/701_dual_lora_epoch_sweep/run_epoch.py"
DATA=/remote_compute/home/data/exp699/solution140_full_support_sources/data.csv
OOF=/remote_compute/home/data/exp699/eval/four_head_oof.npz
MANIFEST="$SOURCE/shared/manifest.tsv.gz"
VENDOR=/remote_compute/home/data/exp699/vendor

run_arm() {
  local gpu=$1 architecture=$2 rank=$3 alpha=$4 model=$5 model_class=$6 cache=$7
  local name="${architecture}_r${rank}_e5"
  local out="$RUN/$name"
  test ! -e "$out"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" EXP_ID=703 CHECKPOINT_EACH_EPOCH=1 \
  ECUP_MODEL_ROOT="$model" ECUP_DATA="$DATA" ECUP_OOF="$OOF" \
  ECUP_MANIFEST="$MANIFEST" ECUP_OUTPUT_DIR="$out/output" ECUP_VENDOR="$VENDOR" \
  HOLDOUT_FOLD=0 TRAINING_MODE=hard MODEL_CLASS="$model_class" \
  USE_CHAT_BATCH="$([ "$model_class" = multimodal ] && echo 1 || echo 0)" \
  EXP701_IMAGE_VIEW="$out/images" PYTORCH_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false "$PY" -u "$WRAPPER" --parent "$PARENT" \
    --architecture "$architecture" --epochs 5 --cache "$cache" \
    --lora-r "$rank" --lora-alpha "$alpha" >"$out/run.log" 2>&1
}

test ! -e "$RUN"
mkdir -p "$RUN"
gpu=0
for rank in 16 32 64; do
  alpha=$((rank * 2))
  run_arm "$gpu" qwen35_4b "$rank" "$alpha" /models/qwen35_4b multimodal "$SOURCE/shared/q35_fold0_cache" &
  echo "$! qwen35_4b $rank" >>"$RUN/owned_pids.txt"
  gpu=$((gpu + 1))
done
for rank in 16 32 64; do
  alpha=$((rank * 2))
  run_arm "$gpu" qwen3vl_2b "$rank" "$alpha" /models/qwen3vl_2b image_text "$SOURCE/shared/q3_fold0_cache" &
  echo "$! qwen3vl_2b $rank" >>"$RUN/owned_pids.txt"
  gpu=$((gpu + 1))
done
wait
