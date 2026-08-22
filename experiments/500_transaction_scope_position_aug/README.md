# 500: position-only transaction-scope augmentation

Статус: **подготовлен, не запускался**. GPU-запуск запрещён до фиксации нового
semantic-family graph и его component-disjoint folds.

## Единственный изменяемый фактор

Эксперимент наследует точный обучающий parent flammable-адаптера решения `400`:
Qwen3.5-4B, hard mode, seed 42, исходный BAD family-diverse selector, обычные
flammable positives и обычные row-level flammable negatives. Parent selector,
row multiplicities, shuffle, optimizer, число epochs/steps, prompt, первое
изображение 448 px и validation scoring не переписаны.

После неизменённого parent selector код рассматривает только выбранные
outer-train строки категории `Легковоспламеняющиеся`. Если description содержит
ровно одно cue-предложение и оно выражает ровно один из закрытых типов отношения
`included`, `excluded` или `compatible`, оно может быть перенесено в начало
description. Если cue sentences несколько, типы конфликтуют, предложение
уже первое или нет безопасного межфразового whitespace, строка остаётся
неизменной.

Целевая вероятность выбора каждой eligible unique row — 50%, без labels и model
scores: первый байт
`SHA-256(version + outer_fold + row_id)` должен быть меньше 128. Training seed в
hash не входит, поэтому conditional seed repeat получает те же transformed rows.
Алгоритм переставляет непересекающиеся byte slices исходной строки и проверяет
равенство character/token multisets. Он не добавляет, не удаляет и не
перефразирует токены. Все occurrences выбранной строки получают одну и ту же
view, но их число не меняется. Поэтому фактическая occurrence-weighted доля не
обязана быть ровно 50%; audit отдельно сообщает unique-row и occurrence-weighted
rates. Outer-validation rows fail closed.

Inference полностью совпадает с `400`: обычный description, один прежний проход,
без parser, дополнительного score, regex override или нового веса.

## Почему это не повторяет 450

`450` удалял контекст и строил отдельный линейный rank по keyword-selected
предложениям. Здесь полный текст сохранён byte-for-byte как мультимножество
токенов, а прежний cross-attentive adapter обучается быть менее чувствительным к
позиции целого evidence sentence. Никакого нового inference-компонента нет.

## Файлы

- `transaction_scope_position_aug.py` — sentence analysis, stable hash,
  byte-preserving movement и plan audit;
- `parent_selector.py` — dependency-light точная копия фиксированной ветки
  parent selector на NumPy/Pandas;
- `build_manifest.py` — строит immutable JSONL manifest и JSON audit без импорта
  torch/transformers;
- `trainer.py` — тонкая обёртка: проверяет audit, патчит только description
  выбранных outer-train строк, сверяет полный порядок records облегчённого
  selector с настоящим parent и передаёт управление неизменённому parent main;
- `local_run.py` — переносимый интерфейс `audit` / `train`;
- `run.py` — стандартный repository experiment runner.

## Локальный audit без обучения

```bash
uv run python experiments/500_transaction_scope_position_aug/local_run.py audit \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --output-dir /path/to/audit-output \
  --fold 0 \
  --seed 42
```

Команда создаёт `augmentation_manifest.jsonl` и `augmentation_audit.json`.
Она сообщает фактические coverage и selection rates как по unique rows, так и с
весом training occurrences.
До любого train нужно проверить:

- parent/candidate record multiset hashes совпадают;
- `multiplicity_unchanged`, `steps_unchanged`, character/token checks истинны;
- outer-validation и BAD transformed counts равны нулю;
- ручной blind audit 200 transformed pairs проходит semantic-preservation gate
  не ниже 98% и agreement `κ ≥ 0.80`.

## Переносимый train interface

Следующая команда является интерфейсом, а не инструкцией на автоматический
запуск; этот пакет её не выполнял:

```bash
uv run --extra vlm python experiments/500_transaction_scope_position_aug/local_run.py train \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --audit-json /path/to/audit-output/augmentation_audit.json \
  --image-manifest /path/to/image_manifest.tsv.gz \
  --images /path/to/image-cache \
  --model-root /path/to/model \
  --vendor /path/to/python-vendor \
  --output-dir /path/to/train-output \
  --fold 0 \
  --seed 42
```

`trainer.py` заново вычисляет plan и до загрузки модели сравнивает version,
outer fold, parent record multiset, manifest hash и plan hash с frozen audit.
Любое расхождение прекращает выполнение.

Стандартный `run.py` также доступен; дополнительные пути передаются через
обычные portable arguments и `--set KEY=VALUE`. Для воспроизводимого опыта
предпочтителен `local_run.py`, так как он явно требует audit.

## Замороженная проверка

Первые два outer folds имеют только право отклонить. На каждом нужны положительный
routed Macro delta, средний delta не ниже `+0.001`, отсутствие роста flammable и
safety false negatives, больше исправлений, чем ухудшений, и byte-identical BAD.
При успехе без изменения parser/hash/recipe считаются все пять folds. Полная
приёмка: Macro `≥ +0.003`, flammable `≥ +0.006`, минимум 4/5 wins,
corrected:regressed `≥ 1.5`, component bootstrap `P(delta>0) ≥ 0.90`, нулевой
рост защитных FN и положительный prior replay. Затем разрешён только repeat seed
31415 на двух screen folds.

Public и sealed holdout не используются для выбора cue patterns, hash threshold,
folds, weights или thresholds.

## Label-blind semantic audit реальных transformed pairs

Проверка `analysis/semantic_audit.json` выполнена по объединению фактически
преобразованных строк manifests folds 0 и 3. Прочитаны только `id`, `name`,
`description` и `category`; labels, predictions и model outputs не открывались.

- fold 0: 39 transformed rows; fold 3: 46; пересечение: 12;
- в объединении: 73 разных строки (`excluded=52`, `included=13`,
  `compatible=8`);
- 73/73 предложений перенесены побайтно, с тем же набором символов и токенов;
- strict semantic pass: **62/73 = 84,93%**;
- strict failures: 11 — четыре предложения содержат несколько scope-отношений,
  три эксплуатационных `без спичек/зажигалки` ошибочно трактуются как условия
  продажи, три переноса оставляют последующую анафору без локального antecedent,
  одна фраза не называет включённый объект без контекста title.

Решение: **NO-GO**. Требование `>=98%` не выполнено, поэтому запуск folds
запрещён. Исправление parser/guard должно стать новой заранее зафиксированной
версией augmentation; текущие manifests нельзя молча переписать.
