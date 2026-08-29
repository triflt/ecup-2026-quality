# 180: Numeric-variant family prior

## Гипотеза

Unanimous BAD wording families с разными числами можно безопасно объединять.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `recurrence_70_30_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/180_fuzzy_family_prior/preprocess.py --data /path/to/data.csv --output-dir /tmp/180_fuzzy_family_prior
python3 experiments/180_fuzzy_family_prior/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/fuzzy_annotator_prior_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/180_fuzzy_family_prior/.local/compute/`.

## Submission

Source directory: `experiments/180_fuzzy_family_prior/submission`. Локальный ZIP строится через `python3 experiments/180_fuzzy_family_prior/build_submission.py --output /tmp/submission.zip`.

## Результат

Donor-only BAD +0.000761; 17/20 recurrence wins; flammable отключён.

Public Macro F1: **0,8923976821**, точная ничья с `140`. Дополнительное правило
не дало измеримой пользы, поэтому эксперимент не продвинут, а production-
чемпионом остаётся более простой `140`.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
