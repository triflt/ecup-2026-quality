# 030: Qwen3-VL embedding classifier

## Гипотеза

Multimodal representation улучшит text baseline.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/030_qwen3vl_embedding/preprocess.py --data /path/to/data.csv --output-dir /tmp/030_qwen3vl_embedding
python3 experiments/030_qwen3vl_embedding/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/qwen-train-code/run_train.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/030_qwen3vl_embedding/.local/compute/`.

## Submission

Source directory: `experiments/030_qwen3vl_embedding/submission`. Локальный ZIP строится через `python3 experiments/030_qwen3vl_embedding/build_submission.py --output /tmp/submission.zip`.

## Результат

Изображения полезны, но random OOF 0.902350 не перенёсся: Public 0.722013.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
