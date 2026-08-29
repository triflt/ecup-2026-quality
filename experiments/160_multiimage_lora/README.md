# 160: Qwen3-VL multi-image LoRA

## Гипотеза

First/second/last images улучшат five-image residual cohort.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `nested_grouped_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/160_multiimage_lora/preprocess.py --data /path/to/data.csv --output-dir /tmp/160_multiimage_lora
python3 experiments/160_multiimage_lora/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/qwen3vl_multiimage_lora_holdout.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/160_multiimage_lora/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Отклонено на untouched fold: дополнительные gallery images размывают first-image signal.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
