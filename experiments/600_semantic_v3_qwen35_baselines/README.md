# 600: Qwen3.5 baseline-компоненты на semantic-family-v3

Статус: **подготовлен, обучение не запускалось; запуск заблокирован до появления
development-only selector scores**.

## Цель

Эксперимент заново строит две Qwen3.5-компоненты route-`400` на новом
`semantic_family_v3`:

- `original` — исходный Qwen3.5 seed `42` из решения `190`;
- `specialist` — семейно-разнообразный BAD-positive selector из `260`, seed `42`.

Для каждой компоненты предусмотрены пять development folds: всего десять
независимых одно-GPU задач. Sealed holdout из 1 853 строк не используется для
обучения, отбора примеров, порогов или оценки. Этот пакет не содержит команды
оценки sealed holdout.

## Физическое исключение sealed holdout

GPU-задача не получает исходный CSV или полное разбиение. Сначала CPU-команда
детерминированно строит scoped runtime-каталог из nested robust-base эксперимента
`601`:

```bash
python experiments/600_semantic_v3_qwen35_baselines/prepare_runtime_inputs.py \
  --data research/data.csv \
  --folds validation/semantic_family_v3/folds.csv \
  --image-manifest research/lora_image_manifest_complete.tsv.gz \
  --robust-base experiments/601_semantic_v3_visual_base_baselines/.local/robust_base_strict_v2/robust_base_semantic_v3.npz \
  --robust-report experiments/601_semantic_v3_visual_base_baselines/.local/robust_base_strict_v2/robust_base_report.json \
  --output-dir experiments/600_semantic_v3_qwen35_baselines/.local/runtime_inputs_strict_v2
```

Эта команда записывает только 11 118 development-строк: CSV, folds, first-image
manifest, selector OOF, provenance и mapping. Presets передают в GPU-задачу
только этот scoped-каталог. Перед импортом parent runner wrapper:

1. проверяет SHA256 и инварианты `validation/semantic_family_v3/folds.csv`;
2. проверяет frozen SHA256 исходного CSV, image manifest и точного parent runner;
3. соединяет folds с исходным CSV по `id` и физически записывает новый CSV только
   из 11 118 development-строк;
4. требует selector-score NPZ ровно с теми же development ID и локальными folds
   `0..4`;
5. сохраняет `id_mapping.csv` с local/original index, development fold и ролью
   `train` или `validation`;
6. записывает audit с нулём sealed-строк во всех физических входах;
7. только после этого импортирует неизменный parent runner.

Runtime wrapper дополнительно проверяет, что selector не вернул ни одной строки
outer validation, и фиксирует multiset hash и точное число optimizer steps по
родительской формуле. После обучения output contract требует predictions ровно
для одного outer development fold и повторно доказывает отсутствие sealed ID.

## Неизменные рецепты

Обе компоненты используют Qwen3.5-4B, multimodal prompt, только первое
изображение, hard selection, rsLoRA, seed `42`, один epoch, batch size `4`,
gradient accumulation `4`, learning rate `2e-4` и исходную policy числа шагов
относительно фактически выбранного development-train multiset.

Для `original` вызывается исходный parent `research/qwen3vl_lora_holdout.py`.
Для `specialist` вызывается
`research/qwen35_bad_family_diverse_positives_lora.py` с включённым только
family-diverse BAD-positive factor. Teacher, pseudo-labels, public feedback,
дополнительные эпохи и full train запрещены.

## Frozen audits

- `analysis/frozen_protocol_manifest.json` фиксирует 10 задач, размеры folds и
  hashes исходных данных и разбиения.
- `analysis/null_selector_parity.json` подтверждает точное совпадение обоих
  selectors на старом протоколе при identity projection: 10 из 10 сравнений
  совпали по порядку records. Этот audit проверяет wrapper, но не разрешает
  применять старые scores в semantic-v3.

## Блокер

В репозитории пока нет selector-score OOF, построенного исключительно на
development split. Старый OOF мог быть обучен с использованием sealed labels и
поэтому запрещён даже как источник hardness для selector.

До запуска robust-base `601` должен завершиться, после чего указанная CPU-команда
должна создать два замороженных selector-файла:

- `development_selector_oof.npz` с ровно 11 118 development ID, folds `0..4` и
  полем `fused_scores` или `fused`;
- `development_selector_provenance.json`, явно подтверждающий, что sealed rows и
  labels не использовались при fit, selection и thresholding, а scores были
  заморожены до Qwen-обучения.

Без них каждая задача завершится до загрузки модели. Это намеренный fail-closed
барьер, а не недостающий optional input.

## Ожидаемый бюджет

План: 10 одно-GPU задач. По историческим Qwen3.5 runs ожидается примерно 35–50
минут на fold после доступности локальных изображений, то есть около 6–8.5
GPU-часов суммарно. При параллельном исполнении всех задач ожидаемое wall time —
около 45–70 минут; последовательный запуск займёт около 6–9 часов. Возможное
повторное скачивание изображений может добавить 10–25 минут на задачу.

Кроме selector-score blocker, перед фактическим запуском необходимо подтвердить
доступность Qwen3.5-4B, исходного CSV, image manifest и достаточной квоты на десять
одно-GPU задач. Ни одна задача этим пакетом не запускалась.
