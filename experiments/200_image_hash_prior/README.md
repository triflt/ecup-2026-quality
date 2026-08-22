# 200: Image-hash prior

## Гипотеза

Exact/perceptual image repeats добавят сигнал поверх text family.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `recurrence_70_30_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/200_image_hash_prior/preprocess.py --data /path/to/data.csv --output-dir /tmp/200_image_hash_prior
python3 experiments/200_image_hash_prior/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/image_hash_prior_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/200_image_hash_prior/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Отклонено: Macro gain только +0.000095, flammable overrides не выбраны.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
