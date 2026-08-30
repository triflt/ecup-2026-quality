# Финальное решение: solution140 + Exp714

Этот каталог — единая точка входа для проверки финальной системы. Solution140
владеет verdict и сохраняет Public Macro F1 `0.8923976821` отправленного
classifier-архива. Exp714 запускается после verdict и генерирует только
комментарий; Public нового объединённого архива пока не измерен.

## Из чего состоит решение

Классификатор объединяет три обученных сигнала и память обучающей выборки:

1. опорный прогноз из TF-IDF и мультимодальных эмбеддингов;
2. LoRA-ветку Qwen3-VL-2B для анализа текста и изображения;
3. LoRA-ветку Qwen3.5-4B для дополнительного анализа контекста;
4. точные совпадения по `id` и нормализованному названию, найденные только в
   обучающей части данных.

После окончательного verdict Qwen3.5-4B explanation-only rsLoRA `r=16`,
`alpha=32` получает поля карточки, image0 448 px и frozen verdict. Он не получает
scores компонентов и не может менять решение. Runtime проверяет длину,
законченность и явное противоречие; невалидный текст заменяется category-aware
fallback.

Веса слияния подбираются отдельно для каждой категории на четырёх внутренних
фолдах и один раз применяются к отложенному внешнему фолду. Отправленный архив
получил Public Macro F1 `0.8923976821312729`. Это значение относится к
неизменившемуся classifier verdict; новый архив Exp714 ожидает Public-проверку.

Краткая история данных и решающих абляций собрана на странице
[`Как мы пришли к решению 140`](../../../docs/research/solution-140-journey.md).

## Карта документов

| Вопрос | Документ или исходник |
|---|---|
| Архитектура | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Обучение и воспроизведение | [`REPRODUCE.md`](REPRODUCE.md) |
| Валидация и локальный лидерборд | [`EVALUATION.md`](EVALUATION.md) |
| Офлайн-запуск, время и размер | [`RUNTIME_AND_SIZE.md`](RUNTIME_AND_SIZE.md) |
| Модели, лицензии и данные | [`MODEL_AND_DATA_CARD.md`](MODEL_AND_DATA_CARD.md) |
| Веса и неизменяемые артефакты | [`artifact-contract.json`](artifact-contract.json) |
| Автоматическая проверка репозитория | [`verify.py`](verify.py) |
| Официальная команда инференса | [`../submission/run.py`](../submission/run.py) |
| Метаданные решения | [`../submission/metadata.json`](../submission/metadata.json) |
| Зафиксированные метрики | [`../results/metrics.json`](../results/metrics.json) |

## Быстрая проверка без весов

```bash
python3 experiments/140_dual_lora_fusion/final/verify.py
pytest -q tests/test_solution_140_repository.py
```

Скрипт дополнительно проверяет, что reasoner подключён только после frozen
solution140 verdict и что formatter не способен изменить mapping `0 → бан`,
`1 → не бан`. Проверка весов включится после публикации файлов и SHA-256.

## Сборка архива после публикации весов

```bash
python experiments/140_dual_lora_fusion/build_submission.py \
  --output /tmp/solution-140.zip
sha256sum /tmp/solution-140.zip
```

Frozen пакет Exp714 записан в [`artifact-contract.json`](artifact-contract.json):
55 770 475 байт, SHA-256
`a0695a55a85ca835d18f23e3700ee3eccc9ba03b9d490653719f474c38861bce`.
Архив загружает пользователь; Public не используется для донастройки.
