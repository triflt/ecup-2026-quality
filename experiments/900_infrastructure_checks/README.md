# 900: Infrastructure and submission checks

## Гипотеза

Schema, runtime и empty-category smokes обнаружат ошибки до leaderboard.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/900_infrastructure_checks/preprocess.py --data /path/to/data.csv --output-dir /tmp/900_infrastructure_checks
python3 experiments/900_infrastructure_checks/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `tools/check_publish_safety.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/900_infrastructure_checks/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Обнаружены и исправлены image-size mismatch, cross-batch OCR contamination и empty-category crash.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
