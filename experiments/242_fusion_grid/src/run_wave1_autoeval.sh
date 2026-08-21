#!/bin/bash
# Auto-run wave-1 eval + fusion when all training adapters are ready (GPUs 0-5 only).
set -e
WORK_DIR=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC
source /home/user/conda/etc/profile.d/conda.sh
conda activate mlitvinov_vllm
cd $WORK_DIR

MARKER=$WORK_DIR/exps/.wave1_autoeval_done
if [ -f "$MARKER" ]; then
  echo "Wave-1 auto-eval already done."
  exit 0
fi

READY=1
for exp in expA01 expA02 expA03 expA04; do
  [ -d "$WORK_DIR/exps/$exp/adapter" ] || { READY=0; echo "Missing $exp/adapter"; }
done
for exp in expB01 expB02; do
  [ -d "$WORK_DIR/exps/$exp/adapter_embed" ] || { READY=0; echo "Missing $exp/adapter_embed"; }
done
if [ "$READY" -ne 1 ]; then
  echo "Wave-1 training not finished yet."
  exit 0
fi

LOCK=$WORK_DIR/exps/.wave1_autoeval.lock
if [ -f "$LOCK" ]; then
  echo "Wave-1 auto-eval already running."
  exit 0
fi
touch "$LOCK"
trap "rm -f $LOCK" EXIT

echo "All wave-1 adapters found. Starting eval + fusion."
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 python3 -u $WORK_DIR/eval_val.py \
  --lora $WORK_DIR/exps/expA01/adapter --dataset noocr --out $WORK_DIR/exps/expA01/eval --gpu 0 --batch 256 &
CUDA_VISIBLE_DEVICES=1 python3 -u $WORK_DIR/eval_val.py \
  --lora $WORK_DIR/exps/expA02/adapter --dataset noocr --out $WORK_DIR/exps/expA02/eval --gpu 1 --batch 256 --max_lora_rank 64 &
CUDA_VISIBLE_DEVICES=2 python3 -u $WORK_DIR/eval_val.py \
  --lora $WORK_DIR/exps/expA03/adapter --dataset ocrlong --out $WORK_DIR/exps/expA03/eval --gpu 2 --batch 256 &
CUDA_VISIBLE_DEVICES=3 python3 -u $WORK_DIR/eval_val.py \
  --lora $WORK_DIR/exps/expA04/adapter --dataset std --out $WORK_DIR/exps/expA04/eval --gpu 3 --batch 256 &

CUDA_VISIBLE_DEVICES=4 python3 -u $WORK_DIR/eval_embedder.py \
  --adapter $WORK_DIR/exps/expB01/adapter_embed --out $WORK_DIR/exps/expB01/eval --gpu 4 --batch_size 4 --n_images 2 &
CUDA_VISIBLE_DEVICES=5 python3 -u $WORK_DIR/eval_embedder.py \
  --adapter $WORK_DIR/exps/expB02/adapter_embed --out $WORK_DIR/exps/expB02/eval --gpu 5 --batch_size 4 --n_images 2 &

wait

mkdir -p $WORK_DIR/exps/fusion
for b in expB01 expB02; do
  for a in expA01 expA02 expA03 expA04; do
    python3 -u $WORK_DIR/run_fusion_grid.py \
      --a_csv $WORK_DIR/exps/$a/eval/val_predictions.csv \
      --b_csv $WORK_DIR/exps/$b/eval/embed_predictions.csv \
      --out $WORK_DIR/exps/fusion/grid_${a}_${b}
  done
done

touch "$MARKER"
echo "Wave-1 auto-eval finished."
