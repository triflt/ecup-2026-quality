# Отчёт о подготовке эксперимента 635

Дата: 2026-08-23. GPU-задачи и оценка данных не запускались.

## Что зафиксировано

- Контроль — полная система `140`, а не компонентный baseline и не route `603`.
- Единственный меняемый сигнал — вероятность Qwen3.5.
- Зафиксированы исходная модель, замена на `632`, равное среднее и строгий
  evidence-gated вариант.
- Веса и production-пороги `140` скопированы без изменений и защищены SHA-256
  исходного production runner.
- Exact/name post-processing применяется одинаково после каждого варианта и
  обязан быть построен donor-only для внешнего fold.
- Screen открывает только folds `0/3`; full принимает только варианты,
  прошедшие screen, и требует победы `4/5`.
- Реализованы Macro/category/fold F1, corrected/regressed, flammable/all-positive
  FN, grouped component bootstrap и runtime gates.
- Проверяемая фраза пересчитывается по `canonical_text + char_start/char_end`;
  доверять готовому флагу из артефакта запрещено.
- Bundle обязан быть label-free, pickle-free и точно совпадать с development
  частью immutable semantic-v3 registry.

## Почему запуск пока запрещён

В репозитории нет checksum-верифицированного OOF replay **полного** 140 на
semantic-v3 с donor-only exact/name правилами. 603 не подходит: это другой
baseline с donor-calibrated thresholds без полного post-processing 140.
Upstream 632 уже прошёл five-fold gate (`+0,0232704`, `4/5`, bootstrap
`P=0,9659`) и больше не является blocker. Единственное недостающее доказательство
— отдельный full-140 replay; компонентный результат 632 не подменяет его.

Evaluator требует оба доказательства и завершает работу ошибкой до чтения
метрик при любом несовпадении. Это сознательный blocker, а не незавершённая
подмена данных.

## Проверки реализации

- `ruff check` — успешно;
- `ruff format --check` — успешно после форматирования;
- unit tests 635 — `7 passed`;
- совместные tests 603/623/624/632/635 — `22 passed`;
- JSON/TOML parsing — успешно;
- `git diff --check` — успешно.
