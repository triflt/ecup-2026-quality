# 230: Second-seed Qwen3.5 ensemble

## Гипотеза

Независимый rsLoRA seed снизит variance и улучшит ensemble.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `nested_grouped_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/230_qwen35_second_seed/preprocess.py --data /path/to/data.csv --output-dir /tmp/230_qwen35_second_seed
python3 experiments/230_qwen35_second_seed/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/qwen35_seed_ensemble_cv.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/230_qwen35_second_seed/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Пять OOF jobs и full-data training завершены; artifacts ещё не агрегированы.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
