# Воспроизведение решения 140

Код запускается из чистого клона репозитория. Для точного повторения
исторических метрик также нужны закрытые обучающие данные, указанные версии
предобученных моделей, два LoRA-адаптера и три зафиксированных OOF-массива. Их
идентификаторы перечислены ниже, бинарные веса будут опубликованы отдельно.

## 1. Окружение и неизменяемые входные данные

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,vlm]'

python tools/audit_dataset.py --data /path/to/data.csv --images /path/to/images
```

Используйте `competition_train_v1` и сохранённый в репозитории файл
`validation/grouped_text_v1/folds.csv`. Не перестраивайте эти фолды на месте.

## 2. Обучение Qwen3-VL-2B rsLoRA на пяти фолдах

```bash
for fold in 0 1 2 3 4; do
  python experiments/110_qwen3vl_lora/run.py \
    --data /path/to/data.csv \
    --images /path/to/images \
    --model-root /path/to/Qwen3-VL-2B-Instruct \
    --output-dir /tmp/solution140/qwen3vl/fold${fold} \
    --set ECUP_MANIFEST=/path/to/lora_image_manifest.tsv.gz \
    --set ECUP_OOF=/path/to/robust_base_oof.npz \
    --set HOLDOUT_FOLD=${fold} \
    --set SEED=42 \
    --set TRAINING_MODE=hard \
    --set MODEL_CLASS=image_text
done
```

## 3. Обучение Qwen3.5-4B rsLoRA на пяти фолдах

```bash
for fold in 0 1 2 3 4; do
  python experiments/130_qwen35_lora/run.py \
    --data /path/to/data.csv \
    --images /path/to/images \
    --model-root /path/to/Qwen3.5-4B \
    --output-dir /tmp/solution140/qwen35/fold${fold} \
    --set ECUP_MANIFEST=/path/to/lora_image_manifest.tsv.gz \
    --set ECUP_OOF=/path/to/robust_base_oof.npz \
    --set HOLDOUT_FOLD=${fold} \
    --set SEED=42 \
    --set TRAINING_MODE=hard \
    --set MODEL_CLASS=multimodal \
    --set USE_CHAT_BATCH=1
done
```

Общий скрипт фиксирует одну эпоху, максимальную длину 1536 токенов,
микробатч 4, накопление градиента 4, seed 42, rsLoRA rank 16, alpha 32,
dropout 0.05 и длинную сторону первого изображения 448 пикселей.

Чтобы обучить два адаптера на всех данных, повторите соответствующие команды с
новыми каталогами вывода и параметром `--set FULL_TRAIN=1`.

## 4. Объединение пяти OOF-файлов каждой модели

```bash
python research/aggregate_lora_oof.py \
  --predictions /tmp/solution140/qwen3vl/fold{0,1,2,3,4}/predictions.csv \
  --base-oof /path/to/robust_base_oof.npz \
  --output /tmp/solution140/qwen3vl_aggregate.json

python research/aggregate_lora_oof.py \
  --predictions /tmp/solution140/qwen35/fold{0,1,2,3,4}/predictions.csv \
  --base-oof /path/to/robust_base_oof.npz \
  --output /tmp/solution140/qwen35_aggregate.json
```

Каждая команда также создаёт файл `.npz`, который использует следующий этап.

## 5. Повторение вложенного слияния

```bash
python experiments/140_dual_lora_fusion/run.py \
  --inputs /tmp/solution140/qwen3vl_aggregate.npz \
           /tmp/solution140/qwen35_aggregate.npz \
  --names qwen3vl qwen35 \
  --step 0.05 \
  --output /tmp/solution140/nested_fusion_report.json
```

Шаг `--step 0.05` нужен, чтобы точно получить зафиксированные веса:

- БАД: опорный прогноз / Qwen3-VL / Qwen3.5 = `0.50 / 0.25 / 0.25`;
- Легковоспламеняющиеся: `0.15 / 0.10 / 0.75`.

## Идентификаторы исторических OOF-артефактов

| Объект | SHA-256 |
|---|---|
| Распределение по фолдам | `03baaa25bd5a3aef6ad94e02067cccda114041f98d7a35a9e06330a425166e4d` |
| OOF опорного прогноза | `5d7467c48fc8a5a73f947f5aa1300071c77ba699b29c12250caf1bcd3176d7ac` |
| Объединённый OOF Qwen3-VL | `ba432e13624e6c3b1c7304ced8cacf580f4ffcc0a0cde1af8b6afb098bf6dc01` |
| Объединённый OOF Qwen3.5, seed 42 | `147f2b2b87d8220566526b38ab0e085c0bc44cd82b0442baca6b019516c3f1d8` |

Последние три массива не хранятся в Git. Для точного повторения исторической
метрики нужно опубликовать сами файлы или неизменяемые ссылки на них. До этого
кодовый путь воспроизводим, но повтор исходного результата по сохранённым
прогнозам остаётся незакрытым.
