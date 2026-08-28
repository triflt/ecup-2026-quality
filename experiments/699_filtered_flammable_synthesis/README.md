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
две эпохи, добавление 10 отрицательных примеров и четырёхкратный вес тех же
десяти положительных примеров.

## ЧТО ПОЛУЧИЛОСЬ

Лучший вариант — Qwen3.5-4B с source v2, positive-only append cap10 и одной
эпохой. Он прошёл frozen full-fivefold с победой на 5/5 folds. Full refit и
production-package smoke выполняются в remote compute. Cap5, cap19, Qwen3-VL-2B,
balanced80, v1-cap40, TF-IDF и nested OOF reweighting закрыты как слабые или
нестабильные.

## СКОЛЬКО РАБОТАЛО

Один Qwen3.5-4B fold занимает 58–73 минуты на одной H100, включая обучение,
inference и запись adapter/predictions contract. Пять cap10 folds были обучены
и проверены; full refit и три коротких двухфолдовых механизма выполняются на
семи H100 параллельно, восьмая зарезервирована под end-to-end package smoke.

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

1. Finish the one-epoch full refit and verify its exact output contract.
2. Build solution 140 with only the Qwen3.5 adapter and its exact preprocessing
   changed, then run the real production pipeline from the generated ZIP.
3. Submit this justified candidate to Public as soon as package smoke passes.
4. Compare the two-epoch, balanced10 and repeat4 twofold screens, and run full
   fivefold only for a mechanism that improves the accepted cap10 profile.

Public data is not used for filtering, training, threshold selection or
ensemble-weight selection.
