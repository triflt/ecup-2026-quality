# 140: слияние двух LoRA-веток

## Гипотеза

Опорный прогноз, Qwen3-VL LoRA и Qwen3.5 LoRA ошибаются по-разному, поэтому их
слияние должно быть устойчивее каждой отдельной ветки.

## Протокол

Данные: `competition_train_v1`. Основной протокол оценки — `nested_grouped_v1`
из `validation/registry.toml`. Зафиксированные групповые фолды лежат в
`validation/grouped_text_v1/folds.csv`. Общий OOF и оценки отдельных фолдов
помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы нужно проверить
`docs/research/foundations/literature-and-competitions.md`, найти сильный
прецедент и объяснить, почему он должен переноситься на наши данные. Цель —
уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

Полный рецепт обучения пяти фолдов и дообучения на всех данных находится в
[`final/REPRODUCE.md`](final/REPRODUCE.md). После получения двух агрегированных
OOF-файлов вложенное слияние запускается так:

```bash
python3 experiments/140_dual_lora_fusion/run.py \
  --inputs /tmp/solution140/qwen3vl_aggregate.npz \
           /tmp/solution140/qwen35_aggregate.npz \
  --names qwen3vl qwen35 \
  --step 0.05 \
  --output /tmp/solution140/nested_fusion_report.json
```

Точка входа — `research/nested_multimodel_fusion.py`. Настройки конкретной
вычислительной платформы должны только вызывать этот скрипт и хранятся локально
в `experiments/140_dual_lora_fusion/.local/compute/`.

## Архив для отправки

Исходники лежат в `experiments/140_dual_lora_fusion/submission`. Локальный ZIP
собирается командой `python3 experiments/140_dual_lora_fusion/build_submission.py --output /tmp/submission.zip`.

## Результат

Основная архитектура получила Macro F1 `0.911843` на вложенной CV-5. Расчётное
время работы укладывается в лимит.

Public Macro F1: **0,8923976821**. Эксперимент `180` получил ровно тот же
результат, поэтому более простой `140` выбран основным решением.
Public не используется для перенастройки весов или порогов.

Подробные машиночитаемые результаты находятся в `results/metrics.json` и
`reports/experiment-log.csv`.

Единая точка входа для жюри и воспроизведения: [`final/`](final/).
