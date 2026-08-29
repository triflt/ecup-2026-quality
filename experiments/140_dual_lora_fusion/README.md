# 140: Dual-LoRA fusion

## Гипотеза

Robust base, Qwen3-VL LoRA и Qwen3.5 LoRA дают устойчивую decision diversity.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `nested_grouped_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

Полный fold-train и full-refit recipe решения 140 находится в
[`final/REPRODUCE.md`](final/REPRODUCE.md). После получения двух агрегированных
OOF-файлов nested fusion запускается так:

```bash
python3 experiments/140_dual_lora_fusion/run.py \
  --inputs /tmp/solution140/qwen3vl_aggregate.npz \
           /tmp/solution140/qwen35_aggregate.npz \
  --names qwen3vl qwen35 \
  --step 0.05 \
  --output /tmp/solution140/nested_fusion_report.json
```

Entrypoint: `research/nested_multimodel_fusion.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/140_dual_lora_fusion/.local/compute/`.

## Submission

Source directory: `experiments/140_dual_lora_fusion/submission`. Локальный ZIP строится через `python3 experiments/140_dual_lora_fusion/build_submission.py --output /tmp/submission.zip`.

## Результат

Главная architecture: nested Macro F1 0.911843; scaled runtime укладывается в лимит.

Public Macro F1: **0,8923976821**. Эксперимент `180` получил ровно тот же
результат, поэтому более простой `140` выбран текущим production-чемпионом.
Public не используется для перенастройки весов или порогов.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.

Единая точка входа для жюри и воспроизведения: [`final/`](final/).
