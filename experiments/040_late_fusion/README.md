# 040: Category-specific late fusion

## Гипотеза

Независимые text и multimodal heads лучше переносятся, чем единая модель.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/040_late_fusion/preprocess.py --data /path/to/data.csv --output-dir /tmp/040_late_fusion
python3 experiments/040_late_fusion/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/group_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/040_late_fusion/.local/compute/`.

## Submission

Source directory: `experiments/040_late_fusion/submission`. Локальный ZIP строится через `python3 experiments/040_late_fusion/build_submission.py --output /tmp/submission.zip`.

## Результат

Подтверждено: grouped Macro F1 0.899297 и лучший Public 0.806579.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
