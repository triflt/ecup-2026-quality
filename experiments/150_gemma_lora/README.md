# 150: Gemma-4-E4B supervised LoRA

## Гипотеза

Gemma добавит независимый multimodal head.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `nested_grouped_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/150_gemma_lora/preprocess.py --data /path/to/data.csv --output-dir /tmp/150_gemma_lora
python3 experiments/150_gemma_lora/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `research/qwen3vl_lora_holdout.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/150_gemma_lora/.local/compute/`.

## Submission

Для этого эксперимента отдельный submission package не формировался.

## Результат

Отклонено после пяти folds: nested fusion 0.902686 и высокая variance flammable.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
