#!/usr/bin/env bash
set -euo pipefail

ROOT=/remote_compute/home/repos/quality
RUN=/remote_compute/home/runs/exp709_retry2
PY=/remote_compute/home/.venv-exp699/bin/python
PARENT="$ROOT/research/qwen3vl_lora_holdout.py"
WRAPPER="$ROOT/experiments/709_gemma4_lora_epoch_sweep/run_epoch.py"
DATA=/remote_compute/home/data/exp699/solution140_full_support_sources/data.csv
OOF=/remote_compute/home/data/exp699/eval/four_head_oof.npz
MANIFEST=/remote_compute/home/runs/exp701/shared/manifest.tsv.gz
VENDOR=/remote_compute/home/data/exp699/vendor
CACHE=/remote_compute/home/runs/exp701/shared/q35_fold0_cache

test ! -e "$RUN"
mkdir -p "$RUN"
: >"$RUN/owned_pids.txt"

launch_arm() {
  local gpu=$1 architecture=$2 rank=$3 alpha=$4 model=$5
  local name="${architecture}_r${rank}"
  local arm="$RUN/$name"
  mkdir -p "$arm"
  CUDA_VISIBLE_DEVICES="$gpu" EXP_ID=709 HOLDOUT_FOLD=0 \
  ECUP_MODEL_ROOT="$model" ECUP_DATA="$DATA" ECUP_OOF="$OOF" \
  ECUP_MANIFEST="$MANIFEST" ECUP_OUTPUT_DIR="$arm/output" ECUP_VENDOR="$VENDOR" \
  TRAINING_MODE=hard MODEL_CLASS=multimodal USE_CHAT_BATCH=1 \
  LINEAR_ONLY_TARGETS=1 \
  EXP709_IMAGE_VIEW="$arm/images" PYTORCH_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false "$PY" -u "$WRAPPER" --parent "$PARENT" \
    --architecture "$architecture" --epochs 5 --cache "$CACHE" \
    --lora-r "$rank" --lora-alpha "$alpha" >"$arm/run.log" 2>&1 &
  echo "$! $gpu $name" | tee -a "$RUN/owned_pids.txt"
}

launch_arm 1 gemma4_e2b 16 32 /tmp/gemma-models/gemma-4-E2B-it
launch_arm 2 gemma4_e2b 32 64 /tmp/gemma-models/gemma-4-E2B-it
launch_arm 3 gemma4_e2b 64 128 /tmp/gemma-models/gemma-4-E2B-it
launch_arm 4 gemma4_e4b 16 32 /tmp/gemma-models/gemma-4-E4B-it
launch_arm 5 gemma4_e4b 32 64 /tmp/gemma-models/gemma-4-E4B-it
launch_arm 6 gemma4_e4b 64 128 /tmp/gemma-models/gemma-4-E4B-it

wait
