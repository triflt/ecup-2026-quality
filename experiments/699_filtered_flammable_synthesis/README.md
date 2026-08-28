# 699: filtered flammable synthesis

## ЧТО ДЕЛАЛ

Эксперимент проверяет добавление небольшого fold-safe набора синтетических
примеров класса «Легковоспламеняющиеся» к полному реальному train-multiset
Qwen3.5-4B. Реальные строки не заменяются и не удаляются. BAD-данные, loss,
seed, порядок реальных данных, effective batch, learning rate, threshold и
frozen production route решения 140 остаются неизменными.

Синтетика отфильтрована отдельно внутри каждого outer fold без чтения его
validation labels. Проверены дозы 5, 10 и 19 положительных synthetic
occurrences. После full-fivefold победы cap10 отдельно проверяются три фактора:
две эпохи, добавление 10 отрицательных примеров и вес тех же десяти
положительных примеров с multiplicity `2` и `4`.

## ЧТО ПОЛУЧИЛОСЬ

Эксперимент закрыт отрицательно. Исторический cap10-кандидат прошёл локальный
frozen full-fivefold с победой на 5/5 folds, но на Public получил
`0.83650126`, то есть `-0.05589642` относительно решения 140. Повторно
отправлять этот ZIP нельзя.

Последующий matched-order тест устранил главный causal confound: synth10
сравнивался с десятью дополнительными real-positive exposures при одинаковых
initialization, 4892 real slots, permutation, RNG, LR trace и 307 optimizer
steps. На fold3 синта не изменила ни одного финального flammable-решения и
ухудшила flammable AP. Fold0 confirmation, новый refit, package и Public
закрыты. Cap5, cap19, Qwen3-VL-2B, balanced80, balanced10, repeat2, repeat4,
v1-cap40, TF-IDF, weighting конфликтов, двухстрочный relabel, nested OOF
reweighting и двухэпохный Qwen3.5 также закрыты.

## СКОЛЬКО РАБОТАЛО

Один Qwen3.5-4B fold занимает 58–73 минуты на одной H100, включая обучение,
inference и запись adapter/predictions contract. Пять cap10 folds были обучены
и проверены. Full refit занял `55.66` минуты на одной H100; production ZIP и
его 8-row end-to-end smoke собраны автоматически после refit. Balanced10 и
repeat2 и repeat4 завершены на folds 0/3. Все GPU screens терминальны; remote compute
остаётся готовым на восьми H100 для следующего направления. Финальный decision
packet дополнительно использовал `11.4710` H100-hours на B/C, weighting,
relabel и matched-order causal arms; matched-order pair заняла `68.20/68.15`
минуты на двух H100.

## МЕТРИКИ

Public:

- solution140 baseline: `0.89239768`;
- `solution140_q35_v2p10_e1_448e0eb4.zip`: `0.83650126`;
- delta: `-0.05589642`;
- ODS filename-to-score binding подтверждён; exact uploaded-bytes-to-local-SHA
  binding остаётся `UNVERIFIED`, потому что hash-bearing upload receipt не был
  сохранён.

Matched-order fold3, synth10 против equal-dose real10, exact frozen route:

- Macro до prior: `0.95270890 → 0.95349078`, delta `+0.00078188`;
- flammable F1: точная ничья `0.92957746`;
- flammable TP/FP/FN: точная ничья `33/4/1`;
- Qwen3.5 flammable AP: `0.92951924 → 0.92608663`, delta `-0.00343261`;
- fused flammable AP: `0.97234051 → 0.96676676`, delta `-0.00557375`;
- final corrections/regressions: `3/0`, но все три относятся к BAD;
- flammable final decision changes: `0`;
- после prior Macro delta `+0.00051364`, flammable F1/FN снова без изменений;
- 41 Qwen3.5 threshold flips дали `0` surviving final threshold flips.

Решение matched-order gate: `REJECT_MATCHED_SYNTH_FOLD3`.

Frozen solution-140 replay, cap10 full fivefold:

- routed Macro F1: `0.91418665 → 0.93362254`, delta `+0.01943588`;
- fold wins: `5/5`;
- flammable F1: `0.87679083 → 0.90909091`;
- flammable TP/FP/FN/TN: `155/16/15/4530`;
- corrections/regressions: `101/26`;
- rare-family corrections/regressions: `67/26`;
- singleton corrections/regressions: `59/21`;
- flammable AP: `0.92497717 → 0.89831246`, delta `-0.02666471`.

Full-refit/package acceptance:

- train occurrences: `5,450 real + 10 synthetic = 5,460`;
- optimizer steps: `342`, final mean loss `0.29048374`;
- package decision: `GO_PUBLIC_SUBMIT`;
- ZIP size: `44,137,261` bytes;
- ZIP SHA-256: `f1976370793e2627225cf38b95d1b9bdbe8832dd2c9801d4afc65b3a2920a4f3`;
- real packaged-pipeline smoke: `8/8` rows, `labels/sealed/Public = 0`.

Two-epoch screen on the exact same folds 0/3 was rejected against the accepted
one-epoch candidate:

- routed Macro delta: `+0.02034326` vs `+0.03941484`;
- flammable F1: `0.86713287` vs `0.90370370`;
- flammable FP/FN: `13/6` vs `6/7`;
- corrections/regressions: `47/12` vs `54/7`;
- AP delta: `+0.03936205` vs `-0.01107883`.

The AP and one-FN improvement do not offset the materially worse Macro/F1,
false-positive count and correction/regression profile.

Balanced10 (10 positive + 10 synthetic negative occurrences) was also rejected
against positive-only repeat1:

- routed Macro delta: `+0.03202746` vs `+0.03941484`;
- flammable F1: `0.88888889` vs `0.90370370`;
- flammable FP/FN: `7/8` vs `6/7`;
- corrections/regressions: `54/9` vs `54/7`;
- AP delta: `+0.01037587` vs `-0.01107883`.

Repeat4 (the same ten positive rows repeated four times) was rejected despite
two fewer FN:

- routed Macro delta: `+0.03757301` vs `+0.03941484`;
- flammable F1: `0.90000000` vs `0.90370370`;
- flammable FP/FN: `9/5` vs `6/7`;
- corrections/regressions: `54/8` vs `54/7`;
- AP delta: `-0.07028599` vs `-0.01107883`.

Repeat2 was clearly rejected:

- routed Macro delta: `+0.01959174` vs `+0.03941484`;
- flammable F1: `0.86567164` vs `0.90370370`;
- flammable FP/FN: `8/10` vs `6/7`;
- corrections/regressions: `48/12` vs `54/7`;
- AP delta: `-0.08631584` vs `-0.01107883`.

AP improvement alone is not treated as a win when Macro/F1/FN are worse.

## ОШИБКИ

Главная ошибка первоначального вывода — считать 5/5 локальных побед causal
доказательством пользы синты. Append менял длину shuffled dataset, optimizer
trajectory и preflight RNG. Public показал сильную инверсию, а matched-order
контроль подтвердил отсутствие улучшения целевой flammable boundary. Поэтому
старые B/C/D метрики считаются descriptive diagnostics, а не ship evidence.

The original handoff recipe and the first local README described replacement
of real flammable rows. That design was rejected. The implemented append arm
proves `augmentation_arm=synth_append` and keeps the full real training
multiset. Evaluator v2 also retuned fusion weights for each candidate and is
invalid; only evaluator schema `exp699_gpu_evaluation_v3` with
`frozen_baseline_140_nested_oof_selection` is authoritative. A nested OOF
weight search was run only after component confirmation, but was inferior to
the frozen solution-140 weights and was not promoted.

The first balanced10 evaluation was blocked before metrics because the frozen
evaluator's technical append-count whitelist omitted the valid value `20`.
The evaluator was extended only to accept exactly `20` (while a negative test
rejects `21`); 31 targeted tests pass. The already trained artifacts were then
evaluated without retraining or any scientific change.

## ЧТО ДАЛЬШЕ

Единственное рекомендованное продолжение — CPU-only family-pure OOF transfer
audit на уже существующих predictions. Строки должны выбираться label-blind по
semantic-family purity так, чтобы одна семья не пересекала train/validation.
Нужно заново сравнить solution140 и доступные exp699 routes по финальным
flammable FP/FN, decision survival, singleton и rare cohorts. До объяснения
local-to-Public inversion новые exp699 training/package/Public запрещены.

Public data is not used for filtering, training, threshold selection or
ensemble-weight selection.
