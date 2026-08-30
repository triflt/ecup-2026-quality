# 130: Qwen3.5 supervised rsLoRA

## Гипотеза

Qwen3.5 даст ошибки, отличные от Qwen3-VL.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `nested_grouped_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/130_qwen35_lora/preprocess.py --data /path/to/data.csv --output-dir /tmp/130_qwen35_lora
python3 experiments/130_qwen35_lora/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/qwen3vl_lora_holdout.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/130_qwen35_lora/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Подтверждено: three-head nested Macro F1 0.911843.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
