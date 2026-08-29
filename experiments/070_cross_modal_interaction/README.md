# 070: Cross-modal interaction head

## Гипотеза

Text/image difference и product terms улучшат сложные BAD cases.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/070_cross_modal_interaction/preprocess.py --data /path/to/data.csv --output-dir /tmp/070_cross_modal_interaction
python3 experiments/070_cross_modal_interaction/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/cross_modal_consistency_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/070_cross_modal_interaction/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Локально улучшает BAD, но advanced submission проигрывает robust late fusion на Public.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
