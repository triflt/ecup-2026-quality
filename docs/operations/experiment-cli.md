# Portable experiment CLI

Каждый `experiments/<id>/run.py` поддерживает одинаковые platform-independent arguments:

```text
--data PATH         → ECUP_DATA
--images PATH       → ECUP_IMAGES
--output-dir PATH   → ECUP_OUTPUT_DIR
--model-root PATH   → ECUP_MODEL_ROOT
--set KEY=VALUE     дополнительный artifact path или hyperparameter
```

Аргументы, не распознанные общим runner, передаются native entrypoint без изменения.

Пример late fusion:

```bash
python3 experiments/040_late_fusion/run.py \
  --data /data/data.csv \
  --set ECUP_ALL_IMAGE_EMBEDDINGS=/data/train_embeddings_fp16.npz \
  --set ECUP_REPORT=/output/group_cv_report.json
```

Пример Qwen3-VL LoRA:

```bash
python3 experiments/110_qwen3vl_lora/run.py \
  --data /data/data.csv \
  --images /data/images \
  --model-root /models/qwen3-vl \
  --output-dir /output \
  --set ECUP_MANIFEST=/data/lora_image_manifest.tsv.gz \
  --set ECUP_OOF=/data/four_head_oof.npz \
  --set HOLDOUT_FOLD=4 \
  --set SEED=42
```

Пример aggregation второго seed:

```bash
python3 experiments/230_qwen35_second_seed/run.py \
  --set ECUP_BASE_OOF=/artifacts/base_oof.npz \
  --set ECUP_QWEN3VL_OOF=/artifacts/qwen3vl_oof.npz \
  --set ECUP_QWEN35_SEED_A_OOF=/artifacts/qwen35_seed42_oof.npz \
  --set ECUP_REPORT=/output/seed_ensemble.json \
  --seed-b-predictions /artifacts/fold0.csv /artifacts/fold1.csv /artifacts/fold2.csv /artifacts/fold3.csv /artifacts/fold4.csv
```

Private scheduler preset должен только доставить эти paths и вызвать ту же команду.
