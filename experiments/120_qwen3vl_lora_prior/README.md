# 120: Qwen3-VL LoRA with product-family prior

## Гипотеза

Train product families повторятся в hidden и сохранят ambiguity annotator process.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `recurrence_70_30_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/120_qwen3vl_lora_prior/preprocess.py --data /path/to/data.csv --output-dir /tmp/120_qwen3vl_lora_prior
python3 experiments/120_qwen3vl_lora_prior/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/annotator_prior_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/120_qwen3vl_lora_prior/.local/compute/`.

## Submission

Source directory: `experiments/120_qwen3vl_lora_prior/submission`. Локальный ZIP строится через `python3 experiments/120_qwen3vl_lora_prior/build_submission.py --output /tmp/submission.zip`.

## Результат

70/30 recurrence simulations положительны; grouped leave-one-out metric имеет transductive leakage и не используется как основной.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
