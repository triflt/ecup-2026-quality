# CV-5: каноническая локальная валидация

Эта папка — единая точка входа в пятифолдовую проверку, на которой сравнивалась основная серия решений. Сами fold assignments остаются в `validation/`, чтобы не ломать пути уже воспроизведённых экспериментов.

## Основной исторический протокол

`grouped_text_v1` содержит 5 category-specific folds для всех train-строк:

- стратификация выполняется внутри категории по бинарной метке;
- строки с одинаковым нормализованным `name + description` группируются;
- seed и число folds зафиксированы;
- каждый объект получает OOF-прогноз ровно один раз;
- threshold, fusion weights и checkpoint selection для внешнего fold выбираются только по остальным folds.

Файлы протокола:

- [`../../validation/grouped_text_v1/folds.csv`](../../validation/grouped_text_v1/folds.csv) — `id`, `category`, `label`, `fold`, `group_hash`;
- [`../../validation/grouped_text_v1/basket.csv`](../../validation/grouped_text_v1/basket.csv) — быстрый screen, который не заменяет полную CV-5;
- [`../../validation/grouped_text_v1/manifest.json`](../../validation/grouped_text_v1/manifest.json) — counts, versions и SHA;
- [`../../validation/build_folds.py`](../../validation/build_folds.py) — воспроизводимый builder.

```bash
python validation/build_folds.py \
  --data /path/to/train.csv \
  --output validation/grouped_text_v1/folds.csv \
  --basket-output validation/grouped_text_v1/basket.csv \
  --manifest-output validation/grouped_text_v1/manifest.json
```

Builder следует запускать только для проверки воспроизводимости или при создании **новой** версии. Существующую раскладку нельзя незаметно перегенерировать: изменение строк, seed, группировки или алгоритма требует нового version ID.

## Более строгая проверка переноса

Историческая CV-5 не полностью изолирует близкие товарные семейства. Поэтому для новых архитектурных выводов дополнительно используется sealed [`semantic_family_v3`](../../validation/semantic_family_v3/): связанные семейства не пересекают development folds, а отдельный holdout не участвует в обычном подборе.

Практическое правило:

1. `grouped_text_v1` — сопоставимость со всей историей и основной OOF replay;
2. `semantic_family_v3` / connected-family guards — проверка устойчивости вывода;
3. быстрый один-два fold screen — только фильтр вычислений;
4. финальный кандидат обязан пройти все пять folds и независимый runtime/schema audit.

## Методологические запреты

- нельзя выбирать threshold или checkpoint по меткам проверяемого fold;
- нельзя менять порядок или состав строк между arms абляции;
- нельзя объявлять single-fold screen полной CV;
- нельзя смешивать результаты разных dataset/evaluation versions в одну дельту;
- нельзя использовать Public как цикл тонкой настройки.

Подробный аудит ограничений и дополнительных gates: [`../../docs/validation.md`](../../docs/validation.md).
