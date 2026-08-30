# 080: Four-head and tree ensemble

## Гипотеза

Нелинейный tree head улучшит редкий flammable class.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/080_four_head_ensemble/preprocess.py --data /path/to/data.csv --output-dir /tmp/080_four_head_ensemble
python3 experiments/080_four_head_ensemble/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/four_head_fusion_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/080_four_head_ensemble/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Обычная grouped CV улучшается, duplicate-stress и Public показывают overfitting.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
