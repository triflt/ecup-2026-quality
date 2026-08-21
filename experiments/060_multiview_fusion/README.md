# 060: First-image and multi-view fusion

## Гипотеза

Первое изображение содержит упаковку и kit evidence, размытые all-images embedding.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/060_multiview_fusion/preprocess.py --data /path/to/data.csv --output-dir /tmp/060_multiview_fusion
python3 experiments/060_multiview_fusion/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/tri_fusion_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/060_multiview_fusion/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Небольшой first-image head улучшает fusion; supervised multi-image LoRA не улучшает first-image LoRA.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
