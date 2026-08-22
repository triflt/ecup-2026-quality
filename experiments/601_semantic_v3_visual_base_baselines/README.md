# 601: базовые компоненты на semantic-family v3

## Статус

`qwen_training_running`. Первый строгий CPU-запуск остановлен до сохранения artifact
из-за подтверждённой ошибки NumPy 2.2.6 на ARM macOS: Accelerate/SME выставляет
ложные floating-point flags при корректном `matmul`. Минимальное воспроизведение
на матрицах из единиц дало три предупреждения при конечном точном результате;
NumPy 2.3.1 на том же входе предупреждений не даёт. Повтор на NumPy 2.3.1
успешно завершён, при этом все RuntimeWarning, ConvergenceWarning и non-finite
проверки остались фатальными. Итог: Macro F1 `0,8935380094`, BAD
`0,9452849741`, «легковоспламеняющиеся» `0,8417910448`. Пять Qwen3-VL folds
запущены 22 августа 2026 года и ожидают результатов.

Эксперимент готовит
два воспроизводимых development-компонента: robust base и точный повтор
Qwen3-VL из эксперимента 110.

Read-only аудит входов завершён со статусом `ready`: порядок всех 12 971 ID
совпадает в данных, разбиении, manifest и обоих наборах эмбеддингов; пересечение
development и sealed равно нулю. Полный машинно-читаемый результат сохранён в
`analysis/input_audit.json`.

Первый локальный robust-base artifact протокола v1 признан недействительным:
при выборе весов inner-validation тексты участвовали в построении vocabulary и
IDF. Consumers принимают только marker
`semantic_v3_robust_base_strict_nested_v2`, поэтому использовать старый artifact
для Qwen selector или метрик невозможно.

Из 12 971 строк только 11 118 строк `development` участвуют в построении моделей.
Все 1 853 строки `sealed_holdout` удаляются до обучения, выбора примеров,
подбора весов и порогов и расчёта метрик. Sealed-часть не открывается этим
экспериментом.

## Robust base

Исторический robust base воспроизводим. Его исходный runner —
`research/four_head_fusion_cv.py`, а text/all-image OOF строился в
`research/build_oof_cache.py`. Сохранены полные all-image и first-image
эмбеддинги с проверяемым отображением ID.

Новый runner сохраняет те же четыре головы и параметры:

- TF-IDF word/char + LinearSVC `C=1`;
- all-image embedding + LinearSVC `C=3`;
- first-image embedding + LinearSVC `C=10`;
- all-image ExtraTrees, 300 деревьев.

Для каждого внешнего fold TF-IDF обучается только на остальных development
строках. Внутри outer-train каждый inner fold имеет собственный TF-IDF:
vocabulary и IDF обучаются только на inner-train, после чего inner-validation
лишь преобразуется. Веса четырёх голов и порог выбираются по этому строгому inner
OOF. Сырые outer-fold scores переводятся в шкалу rank через эмпирическую CDF
inner OOF. Ни текст, ни метки sealed_holdout в этот процесс не входят.

Оба набора image embeddings конечны и уже имеют L2-норму около 1, поэтому
дополнительная нормализация не вводится и исторический recipe сохраняется.
RuntimeWarning, ConvergenceWarning, а также любой non-finite feature,
коэффициент или score приводят к немедленной ошибке вместо сохранения artifact.

```bash
python experiments/601_semantic_v3_visual_base_baselines/train_robust_base.py \
  --output-dir /tmp/sv3-robust-base
```

Ожидаемое время CPU: 25–75 минут в зависимости от числа ядер и памяти.

## Runtime-входы Qwen3-VL

После robust base создаётся физический development-only набор. Он содержит CSV,
first-image manifest, parent OOF и явное отображение исходного ID в development
позицию. Пересечение со sealed ID обязано быть нулевым.

```bash
python experiments/601_semantic_v3_visual_base_baselines/prepare_runtime_inputs.py \
  --robust-base /tmp/sv3-robust-base/robust_base_semantic_v3.npz \
  --output-dir /tmp/sv3-runtime
```

Для пяти готовых presets вместо `/tmp` нужно использовать их замороженные
локальные пути:

```bash
python experiments/601_semantic_v3_visual_base_baselines/train_robust_base.py \
  --output-dir experiments/601_semantic_v3_visual_base_baselines/.local/robust_base
python experiments/601_semantic_v3_visual_base_baselines/prepare_runtime_inputs.py \
  --robust-base experiments/601_semantic_v3_visual_base_baselines/.local/robust_base/robust_base_semantic_v3.npz \
  --output-dir experiments/601_semantic_v3_visual_base_baselines/.local/runtime_inputs
```

Задача получает только два публичных Python-файла, parent runner и отдельный
development-only runtime-архив с обязательной проверкой SHA-256; весь `.local`
целиком не упаковывается.

## Qwen3-VL

Используется точный parent recipe эксперимента 110: seed 42, hard selector,
первое изображение 448, один inference pass, те же LoRA, prompt, batch,
оптимизатор и число эпох. Меняются только данные и внешний fold.

```bash
python experiments/601_semantic_v3_visual_base_baselines/train_qwen3vl_fold.py \
  --fold 0 --runtime-dir /tmp/sv3-runtime --output-dir /tmp/sv3-qwen-f0
```

Аналогично запускаются folds 1–4. В `.local/runtime` лежат пять нейтрально
названных однокарточных presets. Оценка на один fold: 20–40 минут; пять заданий
параллельно — 25–50 минут wall time, последовательно — 1,7–3,4 GPU-часа.

## Контракт компонентов

После завершения пяти folds:

```bash
python experiments/601_semantic_v3_visual_base_baselines/assemble_components.py \
  --robust-base /tmp/sv3-robust-base/robust_base_semantic_v3.npz \
  --qwen-predictions /tmp/sv3-qwen-f{0,1,2,3,4}/lora_holdout_predictions.csv \
  --qwen-reports /tmp/sv3-qwen-f{0,1,2,3,4}/lora_holdout_report.json \
  --output-dir /tmp/sv3-components
```

Bundle содержит только development IDs, fold/category ranks robust base и
Qwen3-VL и порядок компонентов route-400. Qwen3.5 в experiment 601 не строится,
поэтому route-400 целиком здесь не оценивается.

Строгий контракт выходов:

- `robust_base_semantic_v3.npz`: `ids`, `labels`, `categories`, `folds`,
  `semantic_components`, четыре отдельных score-головы, `robust_base_score`,
  `predictions`; строк ровно 11 118, object-массивов нет;
- `lora_holdout_predictions.csv`: только ID одного внешнего fold и колонки
  parent runner, включая `lora_score`; пять файлов вместе покрывают каждый
  development ID ровно один раз;
- `semantic_v3_visual_base_components.npz`: те же development ID в порядке
  `folds.csv`, исходные scores и fold/category ranks для robust base и Qwen3-VL;
- `component_contract.json`: контрольные суммы всех входов и bundle,
  фактические размеры folds и доказательство `sealed_rows_in_outputs = 0`.

`development_id_map.csv` связывает позицию в development-only runtime с
позицией в исходных данных, ID, fold и semantic component. Это отображение
создаётся до запуска GPU и не содержит sealed-строк.

## Проверки

```bash
python experiments/601_semantic_v3_visual_base_baselines/input_audit.py
ruff check experiments/601_semantic_v3_visual_base_baselines \
  tests/test_semantic_v3_visual_base_baselines_601.py
pytest -q tests/test_semantic_v3_visual_base_baselines_601.py
```
