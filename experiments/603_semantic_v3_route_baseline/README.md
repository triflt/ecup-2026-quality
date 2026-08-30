# 603: воспроизводимая route-база на semantic-v3

Статус: **завершён на CPU**. Evaluator проверил все готовые development-only
OOF экспериментов `600`, `601` и `602` и воспроизвёл ровно три заранее
зафиксированных варианта. Sealed holdout не открывался, новые модели не
обучались, GPU не запускались.

## Зачем нужен отдельный evaluator

Метрики одиночного Qwen3.5 из `600/602` не отвечают на вопрос, что произойдёт
внутри полной архитектуры `190/400`: там Qwen3.5 объединяется с robust base и
Qwen3-VL разными весами по категориям. Эксперимент `603` фиксирует новую честную
точку отсчёта на semantic-v3 и запрещает смешивать эффект компонента с иной
калибровкой.

Для каждого outer fold используются неизменённые веса locked-190:

- БАД: `0.50 robust + 0.25 Qwen3-VL + 0.25 Qwen3.5`;
- легковоспламеняющиеся: `0.15 robust + 0.10 Qwen3-VL + 0.75 Qwen3.5`.

Каждый компонент переводится в stable rank отдельно внутри пары
`fold × category`. Единственный threshold категории для проверяемого fold
подбирается точным алгоритмом locked-190 только на остальных четырёх folds. Это
**leave-one-fold-out калибровка поверх уже готовых OOF scores**, а не полностью
end-to-end nested meta-validation: donor OOF score мог быть произведён базовой
моделью, обучавшейся с метками текущего проверяемого fold. Исключение fold из
подбора threshold защищает от прямого использования его меток в калибровке, но
не устраняет этот путь через обучение базовых моделей.

## Ровно три замороженных сравнения

1. `original_route_baseline`: original seed 42 используется в обеих категориях.
2. `category_routed_specialist_400`: original остаётся для БАД, specialist `600`
   используется только для flammable.
3. `fixed_four_seed_probability_mean_route`: sigmoid-вероятности seeds
   `42/31415/271828/161803` усредняются с весом `1/4` до ранжирования и
   подставляются в обе категории.

Другие веса, подмножества seed, thresholds и маршруты evaluator не принимает.
Полный контракт находится в `frozen_protocol.json`.

## Результаты

| Вариант | БАД F1 | Flammable F1 | Macro F1 | Delta к original | Победы folds |
|---|---:|---:|---:|---:|---:|
| Original route | 0,954859 | 0,872404 | **0,913631** | — | — |
| Category route specialist | 0,954859 | 0,858824 | **0,906841** | −0,006790 | 1/5 |
| Fixed four-seed mean | 0,954934 | 0,884273 | **0,919604** | +0,005972 | 3/5 |

Все три значения побитово совпали с заранее записанными reference с точностью
до `1e-12`.

Category route `400` не переносится на более строгую semantic-v3: 6 решений
исправлено и 11 ухудшено. Это согласуется с тем, что его Public-прирост был
маленьким, а старая локальная оценка завышала эффект.

Four-seed mean — лучший из трёх по общей метрике и улучшает прежде всего
flammable на `+0,011869`, но пока **не принимается как кандидат**: выиграно только
`3/5` folds, 29 решений исправлено против 25 ухудшенных, отношение `1,16` ниже
замороженного требования `1,5`, а folds 2 и 4 отрицательны. Результат полезен как
новая база и как сигнал о variance reduction, но не разрешает submission или
подбор состава ансамбля.

## Fail-closed проверки

Evaluator требует:

- complete visual bundle `601` с точной SHA-256;
- ровно 10 каталогов `600`: `original/specialist × 5 folds`;
- ровно 15 каталогов `602`: `3 seeds × 5 folds`;
- валидный self-hash каждого runtime contract;
- совпадение SHA predictions, adapter и доступных audit/report файлов;
- точные IDs, порядок, fold, category и label относительно visual bundle и
  immutable semantic-v3 registry;
- ноль sealed-строк во всех runtime contracts;
- точное совпадение трёх reference-метрик.

При любом расхождении public summary и local predictions не создаются.
Публичный `results/metrics.json` не содержит локальных путей, имён задач или
сырых текстов. Полные predictions сохранены только в `.local/evaluation/`.
Полученные метрики служат воспроизводимой проверкой фиксированных OOF, но не
доказательством полностью независимого end-to-end переноса.

## Воспроизведение

```bash
python experiments/603_semantic_v3_route_baseline/evaluate.py \
  --visual-bundle /path/to/601/semantic_v3_visual_base_components.npz \
  --visual-contract /path/to/601/component_contract.json \
  --exp600-fold-output original 0 /path/to/600/original/fold0 \
  ... ещё 9 записей experiment-600 ... \
  --exp602-seed-output 31415 0 /path/to/602/seed31415/fold0 \
  ... ещё 14 записей experiment-602 ... \
  --output /path/to/new/public_summary.json \
  --local-predictions /path/to/new/route_predictions.npz
```

Precedent — decision-level fusion победителя Rakuten SIGIR 2020: независимые
модальности объединяются маломощной схемой на уровне решений. Здесь перенос
сделан консервативно: веса не переоцениваются, меняется только заранее
объявленный Qwen3.5 component.
