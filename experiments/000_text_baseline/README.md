# 000: TF-IDF text baseline

## Гипотеза

Сильный word+character TF-IDF должен заметно превзойти официальный baseline.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/000_text_baseline/preprocess.py --data /path/to/data.csv --output-dir /tmp/000_text_baseline
python3 experiments/000_text_baseline/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `experiments/000_text_baseline/train.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/000_text_baseline/.local/compute/`.

## Submission

Source directory: `experiments/000_text_baseline/submission`. Локальный ZIP строится через `python3 experiments/000_text_baseline/build_submission.py --output /tmp/submission.zip`.

## Результат

Гипотеза подтверждена, но random holdout оказался оптимистичным; grouped Macro F1 0.889576, Public 0.713026.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
