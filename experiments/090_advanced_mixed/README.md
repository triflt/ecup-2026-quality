# 090: Category-mixed advanced submission

## Гипотеза

Разные лучшие локальные architectures для категорий улучшат Public.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/090_advanced_mixed/preprocess.py --data /path/to/data.csv --output-dir /tmp/090_advanced_mixed
python3 experiments/090_advanced_mixed/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/train_advanced_heads.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/090_advanced_mixed/.local/compute/`.

## Submission

Source directory: `experiments/090_advanced_mixed/submission`. Локальный ZIP строится через `python3 experiments/090_advanced_mixed/build_submission.py --output /tmp/submission.zip`.

## Результат

Отклонено: Public 0.785500 против 0.806579 у simpler late fusion.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
