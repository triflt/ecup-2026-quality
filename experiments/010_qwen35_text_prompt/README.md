# 010: Qwen3.5 text prompting

## Гипотеза

Большая generative model сможет классифицировать карточку по инструкции без обучения.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/010_qwen35_text_prompt/preprocess.py --data /path/to/data.csv --output-dir /tmp/010_qwen35_text_prompt
python3 experiments/010_qwen35_text_prompt/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `experiments/010_qwen35_text_prompt/submission/run.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/010_qwen35_text_prompt/.local/compute/`.

## Submission

Source directory: `experiments/010_qwen35_text_prompt/submission`. Локальный ZIP строится через `python3 experiments/010_qwen35_text_prompt/build_submission.py --output /tmp/submission.zip`.

## Результат

Отклонено: Public 0.579901, существенно ниже supervised text model.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
