# E-CUP 2026 · Quality

Исследовательский репозиторий решения задачи проверки товарных карточек в категориях **«БАД»** и **«Легковоспламеняющиеся»**. Здесь собраны воспроизводимый код, замороженная CV-5, результаты содержательных экспериментов и исходники submission-кандидатов.

> **Статус лучшего решения**
>
> Канонический источник результата — [`reports/champion.json`](reports/champion.json). README намеренно не дублирует меняющуюся метрику вручную.
>
> **Название:** `TODO: финальное название решения`<br>
> **Краткая идея:** `TODO: 2–3 предложения о компонентах и их роли`<br>
> **Почему это лучше:** `TODO: CV-5 / Public / runtime evidence`<br>
> **Что отправляем:** `TODO: точный experiment package и immutable artifact SHA`

## Задача

Для каждой карточки нужно предсказать бинарный вердикт и сформировать конкретное объяснение, согласованное с карточкой. Основная числовая метрика — Macro F1 по двум категориям. Решение работает офлайн и создаёт CSV со столбцами `id` и `result`.

Полная сверка формата, лимитов, моделей и правил синтетических данных находится в [`docs/hackathon/task-and-rules.md`](docs/hackathon/task-and-rules.md).

## Лучшее решение: описание и запуск

Этот раздел — обязательный handoff перед финальной публикацией.

### Архитектура

`TODO: описать входы → модели → fusion/routing → правила → объяснение. Для каждого компонента указать experiment ID и доказанный вклад.`

### Обучение

```bash
# TODO: заменить на одну воспроизводимую команду финального train/refit
python experiments/<BEST_EXPERIMENT>/train.py \
  --data /path/to/train.csv \
  --images /path/to/images \
  --output-dir /path/to/artifacts
```

### Инференс

```bash
# Контракт официального запуска
python experiments/<BEST_EXPERIMENT>/submission/run.py \
  --test_data_path /path/to/test.csv \
  --output-path /path/to/result.csv
```

Перед заменой плейсхолдеров команда должна быть проверена в чистом окружении, а SHA итогового архива — записан в `reports/champion.json`.

## Главная локальная проверка: CV-5

Последние сравнения выполняются на замороженной пятифолдовой схеме. Её единая точка входа — [`datasets/cv5/`](datasets/cv5/):

- [`datasets/cv5/README.md`](datasets/cv5/README.md) — методология, ограничения и правила честного сравнения;
- [`validation/grouped_text_v1/folds.csv`](validation/grouped_text_v1/folds.csv) — исторические fold assignments основной серии моделей;
- [`validation/build_folds.py`](validation/build_folds.py) — скрипт построения раскладки;
- [`validation/semantic_family_v3/`](validation/semantic_family_v3/) — более строгая sealed-проверка переноса по товарным семействам.

Raw train и изображения не публикуются. Все соединения выполняются по `id` с локальной копией данных.

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
| [`validation/`](validation/) | Frozen folds, sealed protocols и integrity manifests |
| [`experiments/`](experiments/) | Воспроизводимые проверки с выводом и финальные кандидаты |
| [`components/`](components/) | Переиспользуемые компоненты решения и их provenance |
| [`reports/`](reports/) | Champion, журнал экспериментов, submissions и hypothesis board |
| [`docs/`](docs/) | Правила, методология, исследования и текущее состояние |
| [`templates/experiment/`](templates/experiment/) | Шаблон нового experiment package |

## Как читать эксперименты

Каждая папка содержит одну проверяемую гипотезу, frozen dataset/evaluation version, команду запуска и machine-readable результат. Отрицательный эксперимент сохраняется, если он действительно был измерен и объясняет, почему ветку не следует повторять. Папки без запуска, результата или самостоятельного вывода в публичную историю не входят.

- [`experiments/README.md`](experiments/README.md) — компактная карта ключевых серий;
- [`reports/experiment-log.csv`](reports/experiment-log.csv) — полный журнал измерений;
- [`reports/hypothesis-board.csv`](reports/hypothesis-board.csv) — активные и закрытые гипотезы;
- [`WHATS_NEXT.md`](WHATS_NEXT.md) — оперативный handoff.

## Соответствие правилам

| Требование | Где обеспечивается |
|---|---|
| Офлайн-инференс | Submission runner не обращается к сети |
| CLI `-i/-o` | Финальный `submission/run.py` принимает официальный контракт |
| CSV `id,result` | Output formatter и schema tests |
| Конкретное объяснение | Отдельный explanation gate перед финальной отправкой |
| Лимиты времени и размера | Runtime smoke + размер immutable archive в champion report |
| Воспроизводимость | Код обучения, configs, manifests и checksums |
| Запрет внешних данных | `datasets/registry.toml` и dataset audit |
| Синтетические данные | Версия генератора, открытая модель и воспроизводимая процедура |

## Что не публикуется

В Git не входят raw competition data, изображения, веса, adapters, embeddings, submission ZIP, credentials, private execution presets, внутренние адреса и служебная переписка. Локальные артефакты размещаются в игнорируемых `artifacts/` и `.local/`.
