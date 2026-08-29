# 220: Beta-smoothed soft family prior

## Гипотеза

Смешивание posterior family probability с model score будет устойчивее hard override.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `recurrence_70_30_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/220_soft_annotator_prior/preprocess.py --data /path/to/data.csv --output-dir /tmp/220_soft_annotator_prior
python3 experiments/220_soft_annotator_prior/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/soft_annotator_prior_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/220_soft_annotator_prior/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Donor-only Macro gain только +0.000330; confirmation random split подготовлен, но не запущен.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
