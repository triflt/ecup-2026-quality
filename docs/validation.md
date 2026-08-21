# Validation protocol

## Frozen outer folds

Основная оценка — пять category-specific `StratifiedGroupKFold` folds с `random_state=42`. Group key строится из нормализованных `name + description`. Все строки одного product family попадают в один fold.

Файлы:

- `validation/registry.toml` — версии evaluation protocols и их связь с dataset version;
- `validation/grouped_text_v1/folds.csv` — `id`, `category`, `label`, `fold`, `group_hash` для всех строк;
- `validation/grouped_text_v1/basket.csv` — frozen outer fold 4 для быстрых и дорогих holdout experiments;
- `validation/grouped_text_v1/manifest.json` — counts, checksums, `data_version` и `evaluation_version`.

Тексты и изображения в validation package не публикуются: исследователь соединяет IDs с локальной копией competition data.

## Nested model selection

Для каждого outer fold:

1. модель/adapter не видит outer validation rows;
2. fusion weights и threshold выбираются только на остальных четырёх folds;
3. выбранная конфигурация применяется один раз к outer fold;
4. итоговый Macro F1 считается по объединённым outer predictions.

Global OOF tuning допускается только как диагностическая верхняя оценка и всегда помечается `optimistic`.

## Дополнительные protocols

- `image-duplicate stress`: объединяет full-text group и exact/near image duplicate group;
- `70/30 recurrence simulation`: оценивает product-family prior при возможном повторении train products в hidden;
- `single-fold basket`: быстрый screen, который не заменяет пять folds;
- Public leaderboard: архитектурная проверка, а не инструмент threshold tuning.

Идентификаторы текущих протоколов: `grouped_text_v1`, `nested_grouped_v1`, `image_duplicate_stress_v1`, `recurrence_70_30_v1`. Изменение состава строк, группировки, seed или selection procedure создаёт новую immutable версию; исторические experiment configs не переписываются.

## Acceptance gate

Эксперимент принимается, если:

- улучшает nested grouped Macro F1 либо является заранее определённым diverse Public candidate;
- не обрушивает ни одну category F1;
- выдерживает stress split или риск явно записан;
- укладывается в runtime и archive limits;
- проходит input/output schema validation;
- train и inference preprocessing идентичны.
