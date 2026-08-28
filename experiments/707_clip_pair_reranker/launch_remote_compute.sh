#!/usr/bin/env bash
set -euo pipefail
ROOT=/remote_compute/home/repos/quality
RUN=/remote_compute/home/runs/exp707_retry1
PY=/remote_compute/home/.venv-exp699/bin/python
BASE=/remote_compute/home/runs/exp699/learned_pair_screen_v1_retry2/preflight.npz
DATA=/remote_compute/home/data/exp699/solution140_full_support_sources/data.csv
IMAGES=/remote_compute/home/runs/exp701/shared/q3_fold0_cache
MODEL=/remote_compute/home/models/clip-vit-base-patch32
PREP="$ROOT/experiments/707_clip_pair_reranker/prepare_clip_packet.py"
TRAIN="$ROOT/experiments/700_qwen_pair_positive_evidence/train_screen.py"
PARENT="$ROOT/experiments/699_filtered_flammable_synthesis/train_learned_pair_screen.py"
PAIR="$ROOT/experiments/699_filtered_flammable_synthesis/evaluate_retrieval_pair_verifier.py"

run_arm() {
  local gpu=$1 mode=$2
  local out="$RUN/$mode"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$PREP" --base-packet "$BASE" \
    --data "$DATA" --image-cache "$IMAGES" --model "$MODEL" --mode "$mode" \
    --output "$out/packet.npz" >"$out/prepare.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$TRAIN" --packet "$out/packet.npz" \
    --parent-trainer "$PARENT" --pair-module "$PAIR" --output "$out/result.json" \
    >"$out/train.log" 2>&1
}

test ! -e "$RUN"
mkdir -p "$RUN"
run_arm 6 multimodal & echo "$! multimodal" >>"$RUN/owned_pids.txt"
run_arm 7 text_only & echo "$! text_only" >>"$RUN/owned_pids.txt"
wait
