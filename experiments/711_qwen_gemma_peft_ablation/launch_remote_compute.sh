#!/usr/bin/env bash
set -euo pipefail

ROOT=/remote_compute/home/repos/quality
RUN=/tmp/exp711-screen-fold0
PY=/remote_compute/home/.venv-exp699/bin/python
PARENT="$ROOT/research/qwen3vl_lora_holdout.py"
WRAPPER="$ROOT/experiments/711_qwen_gemma_peft_ablation/run_epoch.py"
DATA=/remote_compute/home/data/exp699/solution140_full_support_sources/data.csv
OOF=/remote_compute/home/data/exp699/eval/four_head_oof.npz
MANIFEST=/remote_compute/home/runs/exp701/shared/manifest.tsv.gz
VENDOR=/remote_compute/home/data/exp699/vendor
CACHE=/remote_compute/home/data/exp699/q35_solution140_thumbnail448_cache_v1

test ! -e "$RUN"
mkdir -p "$RUN"
: >"$RUN/owned_pids.txt"

launch_arm() {
  local gpu=$1 architecture=$2 method=$3 model=$4
  local name="${architecture}_${method}"
  local arm="$RUN/$name"
  mkdir -p "$arm/output" "$arm/images"
  CUDA_VISIBLE_DEVICES="$gpu" EXP_ID=711 HOLDOUT_FOLD=0 \
  ECUP_MODEL_ROOT="$model" ECUP_DATA="$DATA" ECUP_OOF="$OOF" \
  ECUP_MANIFEST="$MANIFEST" ECUP_OUTPUT_DIR="$arm/output" ECUP_VENDOR="$VENDOR" \
  TRAINING_MODE=hard MODEL_CLASS=multimodal USE_CHAT_BATCH=1 \
  LINEAR_ONLY_TARGETS=1 CHECKPOINT_EACH_EPOCH=1 \
  EXP711_IMAGE_VIEW="$arm/images" PYTORCH_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false "$PY" -u "$WRAPPER" --parent "$PARENT" \
    --architecture "$architecture" --method "$method" --epochs 5 --cache "$CACHE" \
    >"$arm/run.log" 2>&1 &
  echo "$! $gpu $name" | tee -a "$RUN/owned_pids.txt"
}

gpu=0
for architecture in qwen35_4b gemma4_e4b; do
  if [[ "$architecture" == qwen35_4b ]]; then
    model=/models/qwen35_4b
  else
    model=/tmp/gemma-models/gemma-4-E4B-it
  fi
  for method in dora rsdora rspissa rsloraplus; do
    launch_arm "$gpu" "$architecture" "$method" "$model"
    gpu=$((gpu + 1))
  done
done

wait
