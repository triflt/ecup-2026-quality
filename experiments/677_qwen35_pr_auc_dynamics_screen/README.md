# 677 — nested PR-AUC checkpoint dynamics для Qwen3.5-4B

## Статус

`PREPARED_NOT_LAUNCHED`. Runtime построен на CPU и проверен; GPU gate закрыт до терминального
пятиfoldового результата родительского маршрута. Public и sealed rows не используются.

## Конкретный недостаток

Текущий class-only рецепт всегда обучается одну эпоху и сохраняет только последний adapter.
Loss/LR/gradient histories и промежуточных checkpoint нет, поэтому оптимальность последнего шага
не доказана. Для редкого класса это особенно опасно: F1 может ухудшаться после того, как ranking,
измеряемый PR-AUC, уже достиг максимума.

## Один изменяемый фактор

Меняется только доля training trajectory, на которой останавливается модель: `25/50/75/100%`.
Фиксируются:

- Qwen3.5-4B и точная revision;
- binary BCE на разности last-token logits `1−0`;
- bf16, LoRA `r=16`, `alpha=32`, dropout `0.05`, rsLoRA, модули `q/k/v/o`;
- LR `2e-4`, warmup 5%, cosine schedule, AdamW, clip norm 1;
- micro-batch 2, accumulation 8, effective batch 16;
- seed 42, first-image preprocessing и sequence limit;
- один общий training trajectory внутри каждого inner split.

Это намеренно тот же proven single-GPU runtime, что у существующего 4B LoRA. Нет нового training
stack, distributed launch или runtime `pip install`. Обычный ms-swift SFT здесь не является
техническим retry: он заменил бы binary BCE на token cross-entropy и стал бы другой гипотезой.

## Честная nested validation

Outer fold 0 остаётся полностью закрытым. Его training runtime делится на четыре donor-inner
screen по уже замороженным semantic-family folds 1/2/3/4:

- train исключает и outer fold 0, и текущий inner fold;
- validation берётся целиком и остаётся label-free в GPU runtime;
- метки читает только CPU evaluator после получения predictions;
- из каждого train удаляется не больше трёх заранее определённых BAD-negative occurrences, чтобы
  сохранить точную parity исходного micro-batch 4 и proven override micro-batch 2;
- один общий stop fraction выбирается по всем четырём inner folds, а не отдельный checkpoint для
  каждого fold.

Runtime manifest: [`results/runtime_manifest.json`](results/runtime_manifest.json).

## Метрики и gate

Primary: tie-aware flammable Average Precision, эквивалент
`sklearn.metrics.average_precision_score`. Secondary: BAD Average Precision.

Non-final fraction принимается, только если против финального checkpoint:

- выигрывает flammable AP минимум на 3/4 inner folds;
- средний flammable AP gain не меньше `+0.010`;
- BAD AP не падает больше чем на `0.002` ни на одном inner fold.

При успехе выбранный fraction сначала подтверждается новыми outer-fold 0/3 моделями. Только после
положительных обоих outer folds обсуждается перенос на 27B. При неуспехе направление закрывается
без 27B обучения.

## Стоимость

- подготовка runtime: CPU, завершена;
- technical smoke: одна H100, минуты, только после открытия gate;
- полный screen: 4×1 H100 параллельно, ориентир 6–12 GPU-часов суммарно;
- Public submissions: 0;
- лимит одного job: 8 часов с большим operational запасом.

## Почему запуск закрыт

[`results/launch_gate.json`](results/launch_gate.json) остаётся fail-closed. Его можно открыть
только после терминального принятого результата родительского пятиfoldового маршрута и проверки
свободной ёмкости. Это предотвращает конкуренцию с активными jobs и дорогой запуск по слабому
родителю.
