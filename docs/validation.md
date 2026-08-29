# Валидация решения 140

В репозитории есть два разных класса локальной оценки: исторический протокол,
на котором строилось решение 140, и более строгий протокол для новых
архитектурных выводов. Их метрики нельзя смешивать в одной таблице или называть
одной и той же CV.

## Термины без сокращений

### OOF

**OOF (out-of-fold prediction)** — прогноз для train-строки, сделанный моделью,
которая не обучалась на этой строке. В CV-5 данные делятся на пять частей:
модель учится на четырёх и предсказывает пятую; это повторяется пять раз.
Объединение пяти частей даёт один OOF-прогноз для каждой train-строки.

OOF нужен, чтобы честно подбирать fusion и порог: обычный прогноз модели на её
же train-строках был бы слишком оптимистичным. Файл `OOF NPZ` в решении 140 —
это сохранённый массив идентификаторов, folds, labels и таких прогнозов.

### SHA-256

**SHA-256** — 64-символьный цифровой отпечаток файла. Если изменился хотя бы
один байт, SHA становится другим. SHA не скрывает содержимое и не заменяет
лицензию; он связывает отчёт, веса, OOF или submission ZIP с точной неизменной
версией артефакта.

## Какой eval используется сейчас

| Задача | Протокол | Как интерпретировать |
|---|---|---|
| Воспроизвести историческое решение 140 | `grouped_text_v1` + `nested_grouped_v1` | Сопоставимость с основной серией старых экспериментов |
| Решить, продвигать ли новую архитектуру | `semantic_family_v3` | Текущий обязательный leakage-resistant promotion gate |
| Быстро отклонить дорогую гипотезу | заранее зафиксированные 1–2 folds | Только reject-screen, не leaderboard |
| Проверить готовый архив | runtime/schema/offline smoke + Public | Проверка одного immutable кандидата, не настройка |

Канонический локальный рейтинг новых кандидатов:
[`reports/semantic-v3-leaderboard.csv`](../reports/semantic-v3-leaderboard.csv).

## Историческая CV-5 решения 140

`grouped_text_v1` содержит пять зафиксированных category-specific folds.
Одинаковый нормализованный `name + description` старались удерживать в одной
группе. Для каждого outer fold:

1. adapter не видит строки outer fold при обучении;
2. checkpoint, fusion weights и threshold выбираются только по остальным
   четырём folds;
3. конфигурация замораживается и один раз применяется к outer fold;
4. пять outer predictions объединяются в OOF;
5. итоговые category F1 и Macro F1 считаются по объединённым строкам.

Историческая fusion-оценка решения 140 — `0.9118425206` Macro F1. Это значение
записано в
[`experiments/140_dual_lora_fusion/results/metrics.json`](../experiments/140_dual_lora_fusion/results/metrics.json).

Ограничение: поздний аудит нашёл близкие и несколько exact-text семейств,
которые историческая раскладка изолирует не полностью. Поэтому эта CV остаётся
неизменной для воспроизводимости, но не используется как единственное
доказательство новой архитектуры.

## Recurrence simulation — отдельная величина

`0.942878` — историческая оценка полного recipe с train-only exact/name donor
memory в recurrence-oriented simulation. Она отвечает на вопрос «что будет,
если скрытые данные повторяют известные товарные семейства», а не на вопрос о
переносе на новые семьи. Её нельзя выдавать за ту же метрику, что `0.9118425206`
или semantic-v3.

Фактический Public результата решения 140 — `0.8923976821`. Public связан с
точным ZIP через SHA-256 в [`reports/champion.json`](../reports/champion.json).

## Текущий promotion gate: `semantic_family_v3`

`semantic_family_v3` строит связанные товарные семейства и не разрешает одному
семейству пересекать development folds или границу sealed holdout. В протоколе
11 118 development-строк и 1 853 sealed-строки. Разбиение и manifest immutable:

- [`validation/semantic_family_v3/folds.csv`](../validation/semantic_family_v3/folds.csv);
- [`validation/semantic_family_v3/manifest.json`](../validation/semantic_family_v3/manifest.json);
- описание CV-5: [`datasets/cv5/README.md`](../datasets/cv5/README.md);
- полный registry протоколов: [`validation/registry.toml`](../validation/registry.toml).

Для promotion новая ветка должна:

- завершить все пять development folds;
- показать положительную pooled Macro F1 delta;
- выиграть большинство заранее зафиксированных folds;
- не ухудшить flammable F1/FN и не обрушить BAD F1;
- иметь приемлемое отношение corrections к regressions;
- пройти проверку singleton/редких семейств;
- сохранить parity неизменённых компонентов решения 140;
- пройти offline, schema, runtime и size gates.

Sealed holdout открывается только после заморозки кандидата и политики принятия.
Он не используется для обычного перебора гиперпараметров.

## Что нельзя делать

- выбирать checkpoint или threshold по меткам проверяемого fold;
- обучать donor index, prior или selector с использованием outer validation;
- называть one/two-fold screen полной CV-5;
- смешивать `grouped_text_v1`, recurrence и `semantic_family_v3` в одну дельту;
- сравнивать component-only score с full routed system;
- выбирать лучший из множества вариантов по Public;
- менять frozen folds или manifest задним числом;
- считать `SUCCEEDED` достаточным без проверки строк, finite scores, SHA и
  полноты артефактов.

## Воспроизводимость

Кодовый путь решения 140 и точные команды описаны в
[`experiments/140_dual_lora_fusion/final/REPRODUCE.md`](../experiments/140_dual_lora_fusion/final/REPRODUCE.md).
Там же записаны ожидаемые SHA исторических fold/OOF объектов. Сейчас код и
схемы воспроизводимы, но exact numeric replay исходного запуска остаётся
незакрытым до публикации трёх OOF-массивов или immutable ссылок на них.

Изменение данных, seed, группировки или selection procedure создаёт новую
версию evaluation protocol. Старые manifests и результаты не переписываются.
