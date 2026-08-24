# 677 — nested PR-AUC checkpoint dynamics для Qwen3.5-4B

## Статус

`TECHNICAL_SMOKE_PENDING`. Runtime построен на CPU и проверен; терминальный
пятиfoldовый parent gate принят. Сначала разрешён ровно один 1×H100 smoke,
затем четыре inner trajectories только после artifact acceptance. Public и
sealed rows не используются.

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
screen по уже замороженным semantic-family folds 1/2/3/4. **Outer fold 3 в этой схеме не является
слепым:** он участвует в inner selection и влияет на выбор stop fraction.

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

При успехе выбранный fraction разрешено подтверждать только на outer fold 0. Положительный outer0
результат не делает outer3 независимым подтверждением. Для проверки outer3 потребуется отдельный
nested selector, построенный исключительно внутри outer3-train; только после этого можно читать
outer3 labels. При неуспехе на outer0 направление закрывается без 27B обучения.

Преобразование fraction в полный outer0 runtime заморожено до inner selection в
[`results/outer0_step_mapping.json`](results/outer0_step_mapping.json):
`0.25→76`, `0.5→153`, `0.75→230`, `1.0→306` optimizer steps. Будущий confirm обязан
сохранить выбранный и финальный checkpoints из одной training trajectory. Отдельно обученные
selected/final runs не считаются однофакторным сравнением.

## Стоимость

- подготовка runtime: CPU, завершена;
- technical smoke: одна H100, минуты, только после открытия gate;
- полный screen: 4×1 H100 параллельно, ориентир 6–12 GPU-часов суммарно;
- Public submissions: 0;
- лимит одного job: 8 часов с большим operational запасом.

## Launch gate

[`results/launch_gate.json`](results/launch_gate.json) разрешает только один inner-fold-1
technical smoke. Runtime verifier читает вложенный terminal audit 659, проверяет его фактический
SHA, решение, все acceptance gates и покрытие пяти folds. Private preset встраивает фактический
quoted HTTPS URL и до распаковки проверяет immutable bundle SHA-256; это устраняет уже известный
literal-URL transport failure. Сам builder fail-closed запрещает full preset. Полная inner wave
будет открыта отдельным gate только после принятия smoke artifact и повторной проверки свободной
ёмкости.
