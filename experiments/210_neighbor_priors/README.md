# 210: Text and ID neighbour priors

## Гипотеза

Order, character similarity или rare tokens найдут label-consistent neighbours.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `nested_grouped_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/210_neighbor_priors/preprocess.py --data /path/to/data.csv --output-dir /tmp/210_neighbor_priors
python3 experiments/210_neighbor_priors/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/char_tfidf_neighbor_prior_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/210_neighbor_priors/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

ID, character-TFIDF и rare-token variants не дали надёжного nested gain.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
