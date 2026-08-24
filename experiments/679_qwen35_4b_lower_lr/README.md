# 679 — deployable Qwen3.5-4B: learning rate 1e-4

## Статус

`TERMINAL_REJECT_AT_SCREEN`. Оба заранее выбранных screen folds технически
валидны, но научный gate провален. Fold 0 дал Macro F1 `+0.003928`, fold 3 —
`-0.026805`; tie-aware Average Precision для flammable ухудшилась на обоих
folds (`-0.018402` и `-0.026813`). В pooled screen flammable FN выросли на 2,
а corrections/regressions составили `27/42`.

Production replay, confirmation folds `1/2/4`, full-data refit и Public для
этого направления запрещены. 27B model и её adapter не использовались ни при
обучении, ни при инференсе 679.

## Недостаток и механизм

Текущий 4B class-only LoRA обучается одну эпоху с peak LR `2e-4`, сохраняет
только финальный checkpoint и не пишет loss/checkpoint dynamics. Оптимальность
peak LR поэтому не доказана. Более низкий LR `1e-4` должен уменьшить
переобучение редкого класса и сохранить ranking, измеряемый tie-aware Average
Precision, без изменения production architecture.

## Единственный изменяемый фактор

Меняется только peak learning rate:

- control: `2e-4`;
- candidate: `1e-4`.

Не меняются model/revision, binary BCE, одна эпоха, 306 optimizer updates,
warmup 5%, cosine schedule, AdamW, clip norm, bf16, seed 42, LoRA
`r=16/alpha=32/dropout=0.05/rsLoRA/q-k-v-o`, sampling, first-image input,
micro-batch 2, accumulation 8, effective batch 16, threshold 0 и inference
route. Runtime берётся byte-identical из принятого 4B control.

## Валидация

Screen folds `0/3` заморожены до запуска. Primary metric для flammable —
tie-aware Average Precision, эквивалентная
`sklearn.metrics.average_precision_score`. Threshold и weights не подбираются.

Screen открывает folds `1/2/4`, только если:

- flammable AP положителен на обоих folds и средний gain `>= +0.005`;
- Macro F1 положителен на обоих folds и средний gain `>= +0.0015`;
- BAD AP не падает больше `0.002` ни на одном fold;
- pooled category F1 не падает больше `0.002`;
- corrections/regressions `>= 1.5` и flammable FN не растёт.

Полный кандидат принимается для refit только если folds `1/2/4` все
положительны по Macro, минимум 4/5 folds положительны, mean Macro gain
`>= +0.006`, flammable F1 gain `>= +0.012`, flammable AP gain положителен
минимум на 4/5 folds и mean `>= +0.005`, BAD guard соблюдён,
corrections/regressions `>= 1.5`, FN не растёт и semantic-singleton slice
положителен.

Standalone-метрики не дают права на запуск confirmation или refit. После
каждого standalone gate тот же кандидат проверяется в checksum-locked
semantic-v3 replay полной production fusion. В production меняется только
Qwen3.5-сигнал для flammable; BAD использует исходный сигнал побайтово, а
robust-base, Qwen3-VL, rank transform, веса, пороги и donor-only exact/name
rules остаются неизменными.

Production screen обязан выиграть folds `0/3`, дать средний fold Macro gain
`>=+0.0015`, corrections/regressions `>=1.5` и не увеличить FN. Полный
production gate требует положительные folds `1/2/4`, не менее `4/5` побед,
Macro gain `>=+0.006`, flammable F1 gain `>=+0.012` и grouped component
bootstrap `P(gain>0)>=0.90`. Confirmation gate криптографически связан с SHA
обоих принятых screen reports.

## Стоимость и deployability

- screen: `2x1 H100`, ожидаемо 45–60 минут wall time;
- confirmation: `3x1 H100`, ожидаемо 45–60 минут;
- full-data refit: `1x1 H100`, ожидаемо до 75 минут;
- production base: только Qwen3.5-4B;
- Public: одна отправка только после full CV, refit и runtime smoke.

Любой отрицательный screen fold или техническое изменение recipe закрывает
направление без controlled scientific retry.
