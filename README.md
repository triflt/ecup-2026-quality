# E-CUP 2026 · Quality

Исследовательский репозиторий решения задачи проверки товарных карточек в категориях **«БАД»** и **«Легковоспламеняющиеся»**. Здесь собраны воспроизводимый код, замороженная CV-5, результаты содержательных экспериментов и исходники кандидатов на отправку.

> **Статус решения 140**
>
> Канонический источник результата — [`reports/champion.json`](reports/champion.json). README намеренно не дублирует меняющуюся метрику вручную.
>
> **Название:** `Решение 140 — TODO: финальное публичное название`<br>
> **Краткая идея:** `TODO: 2–3 предложения о компонентах и их роли`<br>
> **Почему это лучше:** `TODO: CV-5 / Public / замеры времени работы`<br>
> **Что отправляем:** `TODO: точный пакет решения и SHA неизменяемого архива`

## Задача

Для каждой карточки нужно предсказать бинарный вердикт и сформировать конкретное объяснение, согласованное с карточкой. Основная числовая метрика — Macro F1 по двум категориям. Решение работает офлайн и создаёт CSV со столбцами `id` и `result`.

Полная сверка формата, лимитов, моделей и правил синтетических данных находится в [`docs/hackathon/task-and-rules.md`](docs/hackathon/task-and-rules.md).
Текущая оценка готовности репозитория по этим требованиям — в
[`docs/hackathon/repository-readiness.md`](docs/hackathon/repository-readiness.md).

## Лучшее решение: описание и запуск

Этот раздел — точка входа перед финальной публикацией.

Технический пакет решения 140 уже собран в
[`experiments/140_dual_lora_fusion/final/`](experiments/140_dual_lora_fusion/final/):
архитектура, точные команды обучения, протоколы оценки, замеры времени и размера,
модели, данные, контракт артефактов и автоматическая проверка.

Визуальная история исследования — от структуры данных и сильной базовой модели
до слияния двух LoRA-веток — собрана на одной странице:
[`Как мы пришли к решению 140`](docs/research/solution-140-journey.md).

### Архитектура

Решение объединяет TF-IDF по тексту, Qwen3-VL Embedding по тексту и всем
изображениям, а также две rsLoRA-ветки по тексту и первому изображению. Для БАД
и легковоспламеняющихся товаров используются отдельные веса моделей. После
слияния применяется память обучающей выборки: она исправляет только точные
совпадения по `id` или нормализованному названию.

### Обучение

```bash
# Пять внешних фолдов Qwen3-VL-2B.
for fold in 0 1 2 3 4; do
  python3 experiments/110_qwen3vl_lora/run.py \
    --data /path/to/data.csv --images /path/to/images \
    --model-root /path/to/Qwen3-VL-2B-Instruct \
    --output-dir /tmp/solution140/qwen3vl/fold${fold} \
    --set ECUP_MANIFEST=/path/to/lora_image_manifest.tsv.gz \
    --set ECUP_OOF=/path/to/robust_base_oof.npz \
    --set HOLDOUT_FOLD=${fold} --set SEED=42 \
    --set TRAINING_MODE=hard --set MODEL_CLASS=image_text
done

# Пять внешних фолдов Qwen3.5-4B.
for fold in 0 1 2 3 4; do
  python3 experiments/130_qwen35_lora/run.py \
    --data /path/to/data.csv --images /path/to/images \
    --model-root /path/to/Qwen3.5-4B \
    --output-dir /tmp/solution140/qwen35/fold${fold} \
    --set ECUP_MANIFEST=/path/to/lora_image_manifest.tsv.gz \
    --set ECUP_OOF=/path/to/robust_base_oof.npz \
    --set HOLDOUT_FOLD=${fold} --set SEED=42 \
    --set TRAINING_MODE=hard --set MODEL_CLASS=multimodal \
    --set USE_CHAT_BATCH=1
done

# После объединения OOF-прогнозов (точные команды — в final/REPRODUCE.md):
python3 experiments/140_dual_lora_fusion/run.py \
  --inputs /path/to/qwen3vl_aggregate.npz /path/to/qwen35_aggregate.npz \
  --names qwen3vl qwen35 \
  --step 0.05 \
  --output /path/to/nested_fusion_report.json
```

### Инференс

```bash
# Контракт официального запуска
python3 experiments/140_dual_lora_fusion/submission/run.py \
  -i /path/to/test.csv \
  -o /path/to/result.csv
```

Перед заменой плейсхолдеров команду нужно проверить в чистом окружении, а SHA итогового архива записать в `reports/champion.json`.

## Главная локальная проверка: CV-5

Последние сравнения выполняются на замороженной пятифолдовой схеме. Её единая точка входа — [`datasets/cv5/`](datasets/cv5/):

- [`datasets/cv5/README.md`](datasets/cv5/README.md) — методология, ограничения и правила честного сравнения;
- [`validation/grouped_text_v1/folds.csv`](validation/grouped_text_v1/folds.csv) — историческое распределение строк по фолдам;
- [`validation/build_folds.py`](validation/build_folds.py) — скрипт построения раскладки;
- [`validation/semantic_family_v3/`](validation/semantic_family_v3/) — обязательная проверка новых архитектур перед их принятием.

Исторические результаты решения 140 сравниваются на `grouped_text_v1`; новые
кандидаты — только на полном `semantic_family_v3`. Эти таблицы нельзя смешивать.
Канонический текущий локальный лидерборд: [`reports/semantic-v3-leaderboard.csv`](reports/semantic-v3-leaderboard.csv).

Исходная обучающая выборка и изображения не публикуются. Все соединения выполняются по `id` с локальной копией данных.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

python tools/audit_dataset.py
python tools/list_experiments.py
python validation/build_folds.py --data /path/to/train.csv
```

## Структура репозитория

| Путь | Что внутри |
|---|---|
| [`datasets/`](datasets/) | Реестр данных и каноническая точка входа в CV-5 |
| [`validation/`](validation/) | Зафиксированные фолды, закрытые протоколы и манифесты целостности |
| [`experiments/`](experiments/) | Воспроизводимые проверки с выводом и финальные кандидаты |
| [`components/`](components/) | Переиспользуемые компоненты решения и сведения об их происхождении |
| [`reports/`](reports/) | Лучший результат, журнал экспериментов, отправки и локальный лидерборд |
| [`docs/`](docs/) | Правила, методология, исследования и текущее состояние |
| [`templates/experiment/`](templates/experiment/) | Шаблон нового эксперимента |

## Как читать эксперименты

Каждая папка содержит одну проверяемую гипотезу, зафиксированные версии данных и оценки, команду запуска и машиночитаемый результат. Отрицательный эксперимент сохраняется, если он действительно был измерен и объясняет, почему ветку не следует повторять. Папки без запуска, результата или самостоятельного вывода в публичную историю не входят.

- [`experiments/README.md`](experiments/README.md) — компактная карта ключевых серий;
- [`reports/experiment-log.csv`](reports/experiment-log.csv) — полный журнал измерений;
- [`docs/research/archive/removed-experiments-through-652.md`](docs/research/archive/removed-experiments-through-652.md) — краткие выводы удалённых незапущенных и терминально отклонённых веток;
- [`docs/current-status.md`](docs/current-status.md) — компактное текущее состояние без оперативных черновиков.

## Соответствие правилам

| Требование | Где обеспечивается |
|---|---|
| Офлайн-инференс | Скрипт отправляемого решения не обращается к сети |
| CLI `-i/-o` | Финальный `submission/run.py` принимает официальный контракт |
| CSV `id,result` | Форматирование вывода и тесты схемы |
| Конкретное объяснение | Отдельная проверка объяснений перед финальной отправкой |
| Лимиты времени и размера | Контрольный запуск и размер неизменяемого архива в отчёте лучшего решения |
| Воспроизводимость | Код обучения, конфигурации, манифесты и контрольные суммы |
| Запрет внешних данных | `datasets/registry.toml` и аудит данных |
| Синтетические данные | Версия генератора, открытая модель и воспроизводимая процедура |

## Что не публикуется

В Git не входят исходные данные соревнования, изображения, веса, адаптеры,
эмбеддинги, ZIP-архивы отправок, учётные данные, внутренние настройки запуска,
служебные адреса и переписка. Локальные артефакты размещаются в игнорируемых
каталогах `artifacts/` и `.local/`.
