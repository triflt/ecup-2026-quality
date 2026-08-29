#!/usr/bin/env bash
set -euo pipefail
ROOT=/remote_compute/home/repos/quality
RUN=/remote_compute/home/runs/exp708
PY=/remote_compute/home/.venv-exp699/bin/python
PARENT="$ROOT/research/qwen3vl_lora_holdout.py"
WRAPPER="$ROOT/experiments/701_dual_lora_epoch_sweep/run_epoch.py"
DATA=/remote_compute/home/data/exp699/solution140_full_support_sources/data.csv
OOF=/remote_compute/home/data/exp699/eval/four_head_oof.npz
MANIFEST=/remote_compute/home/runs/exp701/shared/manifest.tsv.gz
VENDOR=/remote_compute/home/data/exp699/vendor
CACHE=/remote_compute/home/data/exp699/q35_solution140_thumbnail448_cache_v1

test ! -e "$RUN"
mkdir -p "$RUN"
CUDA_VISIBLE_DEVICES=0 EXP_ID=708 FULL_TRAIN=1 \
ECUP_MODEL_ROOT=/models/qwen35_4b ECUP_DATA="$DATA" ECUP_OOF="$OOF" \
ECUP_MANIFEST="$MANIFEST" ECUP_OUTPUT_DIR="$RUN/output" ECUP_VENDOR="$VENDOR" \
TRAINING_MODE=hard MODEL_CLASS=multimodal USE_CHAT_BATCH=1 \
EXP701_IMAGE_VIEW="$RUN/images" PYTORCH_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false "$PY" -u "$WRAPPER" --parent "$PARENT" \
  --architecture qwen35_4b --epochs 3 --cache "$CACHE" \
  --lora-r 32 --lora-alpha 64 >"$RUN/run.log" 2>&1
