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

Лучший вариант — Qwen3.5-4B с source v2, positive-only append cap10 и одной
эпохой. Он прошёл frozen full-fivefold с победой на 5/5 folds. Full refit и
production-package smoke завершились успешно; submission ZIP готов к Public
upload. Cap5, cap19, Qwen3-VL-2B,
balanced80, v1-cap40, TF-IDF и nested OOF reweighting закрыты как слабые или
нестабильные.

## СКОЛЬКО РАБОТАЛО

Один Qwen3.5-4B fold занимает 58–73 минуты на одной H100, включая обучение,
inference и запись adapter/predictions contract. Пять cap10 folds были обучены
и проверены. Full refit занял `55.66` минуты на одной H100; production ZIP и
его 8-row end-to-end smoke собраны автоматически после refit. Четыре
двухфолдовых механизма продолжают работать параллельно на восьми H100.

## МЕТРИКИ

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

AP improvement alone is not treated as a win when Macro/F1/FN are worse.

## ОШИБКИ

The original handoff recipe and the first local README described replacement
of real flammable rows. That design was rejected. The implemented append arm
proves `augmentation_arm=synth_append` and keeps the full real training
multiset. Evaluator v2 also retuned fusion weights for each candidate and is
invalid; only evaluator schema `exp699_gpu_evaluation_v3` with
`frozen_baseline_140_nested_oof_selection` is authoritative. A nested OOF
weight search was run only after component confirmation, but was inferior to
the frozen solution-140 weights and was not promoted.

## ЧТО ДАЛЬШЕ

1. Transfer the accepted ZIP to the local ignored upload directory after the
   required explicit corporate-artifact download approval.
2. Submit this justified candidate to Public after the action-time confirmation.
3. Compare the two-epoch, balanced10, repeat2 and repeat4 twofold screens, and
   run full fivefold only for a mechanism that improves the accepted cap10
   profile.

Public data is not used for filtering, training, threshold selection or
ensemble-weight selection.
