# 190: Rare-shingle neighbour prior

## Гипотеза

Rare five-word shingles найдут почти одинаковые BAD cards без exact match.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `recurrence_70_30_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/190_shingle_neighbor_prior/preprocess.py --data /path/to/data.csv --output-dir /tmp/190_shingle_neighbor_prior
python3 experiments/190_shingle_neighbor_prior/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/shingle_neighbor_prior_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/190_shingle_neighbor_prior/.local/compute/`.

## Submission

Source directory: `experiments/190_shingle_neighbor_prior/submission`. Локальный ZIP строится через `python3 experiments/190_shingle_neighbor_prior/build_submission.py --output /tmp/submission.zip`.

## Результат

Donor-only BAD +0.000853; 20/20 проверок повторяемости выиграны. Полная система получила **0.8919244237 Public Macro F1**. Этот результат подтверждает архитектуру целиком, но сам по себе не измеряет отдельный вклад shingle prior.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
