# 050: Duplicate-aware stress validation

## Гипотеза

Image duplicates объясняют завышение обычной grouped CV.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `image_duplicate_stress_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/050_duplicate_stress/preprocess.py --data /path/to/data.csv --output-dir /tmp/050_duplicate_stress
python3 experiments/050_duplicate_stress/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/image_duplicate_group_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/050_duplicate_stress/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Подтверждено: ExtraTrees теряет вес, Macro F1 снижается до 0.900610.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
