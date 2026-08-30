# E-CUP 2026 · Quality

Решение задачи модерации товарных карточек в категориях **«БАД»** и **«Легковоспламеняющиеся»**. Репозиторий содержит код решения 140, честную CV-5, журнал экспериментов и воспроизводимый пакет для запуска.

> **Решение 140:** Public Macro F1 — **0.8923976821** · полностью офлайн · [архитектура и история](docs/research/solution-140-journey.md) · [пакет воспроизведения](experiments/140_dual_lora_fusion/final/)

## Задача

Для каждой карточки нужно определить вердикт **«бан / не бан»** и сформировать конкретное объяснение. Основная метрика — Macro F1 по двум категориям; официальный формат и ограничения описаны в [правилах задачи](docs/hackathon/task-and-rules.md).

## Решение 140

Опорный прогноз строят TF-IDF и Qwen3-VL Embedding по тексту и всем изображениям. Две rsLoRA-модели — Qwen3-VL-2B и Qwen3.5-4B — обрабатывают текст и первое изображение. Их прогнозы объединяются с отдельными весами для каждой категории, после чего точные совпадения по `id` или нормализованному названию корректирует память обучающей выборки.

Подробнее: [как мы пришли к решению 140](docs/research/solution-140-journey.md) и [полная спецификация](experiments/140_dual_lora_fusion/final/README.md).

## Запуск

```bash
python3 experiments/140_dual_lora_fusion/submission/run.py \
  -i /path/to/test.csv \
  -o /path/to/result.csv
```

Обучение и точное воспроизведение описаны в [`final/REPRODUCE.md`](experiments/140_dual_lora_fusion/final/REPRODUCE.md). Проверка контракта репозитория:

```bash
python3 experiments/140_dual_lora_fusion/final/verify.py
```

## Валидация

Исторические результаты решения 140 получены на `grouped_text_v1`. Новые архитектурные выводы принимаются только после проверки на более строгом `semantic_family_v3`; результаты этих протоколов не смешиваются.

- [методология CV-5](datasets/cv5/README.md)
- [актуальный локальный лидерборд](reports/semantic-v3-leaderboard.csv)
- [полное описание валидации](docs/validation.md)

## Навигация

| Что искать | Где |
|---|---|
| Финальное решение | [`experiments/140_dual_lora_fusion/final/`](experiments/140_dual_lora_fusion/final/) |
| История исследования | [`docs/research/solution-140-journey.md`](docs/research/solution-140-journey.md) |
| Карта экспериментов | [`experiments/README.md`](experiments/README.md) |
| Журнал измерений | [`reports/experiment-log.csv`](reports/experiment-log.csv) |
| Готовность репозитория | [`docs/hackathon/repository-readiness.md`](docs/hackathon/repository-readiness.md) |

## Данные и артефакты

Исходные данные, изображения и веса не публикуются. Код, конфигурации, схемы артефактов и контрольные суммы находятся в репозитории; локальные бинарные файлы размещаются в игнорируемых каталогах `artifacts/` и `.local/`.
