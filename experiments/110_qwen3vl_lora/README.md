# 110: Qwen3-VL supervised rsLoRA

## Гипотеза

Domain adaptation первого изображения улучшит representation без полного fine-tuning.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `nested_grouped_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/110_qwen3vl_lora/preprocess.py --data /path/to/data.csv --output-dir /tmp/110_qwen3vl_lora
python3 experiments/110_qwen3vl_lora/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/qwen3vl_lora_holdout.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/110_qwen3vl_lora/.local/compute/`.

## Submission

Source directory: `experiments/110_qwen3vl_lora/submission`. Локальный ZIP строится через `python3 experiments/110_qwen3vl_lora/build_submission.py --output /tmp/submission.zip`.

## Результат

Nested grouped Macro F1 fusion 0.906395, +0.00710 к robust base.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
