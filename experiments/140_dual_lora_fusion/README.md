# 140: Dual-LoRA fusion

## Гипотеза

Robust base, Qwen3-VL LoRA и Qwen3.5 LoRA дают устойчивую decision diversity.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `nested_grouped_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/140_dual_lora_fusion/preprocess.py --data /path/to/data.csv --output-dir /tmp/140_dual_lora_fusion
python3 experiments/140_dual_lora_fusion/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/nested_multimodel_fusion.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/140_dual_lora_fusion/.local/compute/`.

## Submission

Source directory: `experiments/140_dual_lora_fusion/submission`. Локальный ZIP строится через `python3 experiments/140_dual_lora_fusion/build_submission.py --output /tmp/submission.zip`.

## Результат

Главная architecture: nested Macro F1 0.911843; scaled runtime укладывается в лимит.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
