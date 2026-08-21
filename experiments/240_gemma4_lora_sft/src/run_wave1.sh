#!/bin/bash
# Wave 1: 6 experiments on GPUs 0-5 (GPU 6-7 are busy with other users' jobs!)
set -e
WORK_DIR=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC
source /home/user/conda/etc/profile.d/conda.sh
conda activate mlitvinov_vllm
cd $WORK_DIR

for exp in expA01 expA02 expA03 expA04 expB01 expB02; do
  mkdir -p $WORK_DIR/exps/$exp
done
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 nohup python3 -u train_lora.py \
  --exp_id expA01 --dataset noocr --r 64 --alpha 64 --lr 1e-4 --epochs 2 \
  --batch_size 4 --grad_accum 8 --gpu 0 \
  > $WORK_DIR/exps/expA01/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 -u train_lora.py \
  --exp_id expA02 --dataset noocr --r 32 --alpha 32 --lr 1e-4 --epochs 2 \
  --batch_size 4 --grad_accum 8 --gpu 1 \
  > $WORK_DIR/exps/expA02/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python3 -u train_lora.py \
  --exp_id expA03 --dataset ocrlong --r 64 --alpha 64 --lr 1e-4 --epochs 2 \
  --batch_size 4 --grad_accum 8 --gpu 2 \
  > $WORK_DIR/exps/expA03/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python3 -u train_lora.py \
  --exp_id expA04 --dataset std --r 64 --alpha 64 --lr 1e-4 --epochs 2 \
  --batch_size 4 --grad_accum 8 --gpu 3 \
  > $WORK_DIR/exps/expA04/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python3 -u train_contrastive.py \
  --exp_id expB01 --r 16 --alpha 16 --lr 1e-4 --epochs 3 \
  --batch_size 8 --grad_accum 4 --samples_per_class 2 \
  --loss_type sigmoid --tau 0.07 --n_images 2 --gpu 4 \
  > $WORK_DIR/exps/expB01/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python3 -u train_contrastive.py \
  --exp_id expB02 --r 16 --alpha 16 --lr 1e-4 --epochs 3 \
  --batch_size 8 --grad_accum 4 --samples_per_class 2 \
  --loss_type triplet --margin 0.3 --n_images 2 --gpu 5 \
  > $WORK_DIR/exps/expB02/train.log 2>&1 &

echo "Wave 1 (GPU 0-5) launched. Monitor: tail -f $WORK_DIR/exps/exp*/train.log"
