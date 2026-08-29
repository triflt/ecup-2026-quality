# 100: Targeted OCR hard-case correction

## Гипотеза

OCR первого изображения исправит uncertainty/disagreement cases.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/100_paddleocr_hard_cases/preprocess.py --data /path/to/data.csv --output-dir /tmp/100_paddleocr_hard_cases
python3 experiments/100_paddleocr_hard_cases/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/paddleocr_extract_hard.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/100_paddleocr_hard_cases/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Помогает слабому flammable baseline, но не складывается с сильной LoRA fusion.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
