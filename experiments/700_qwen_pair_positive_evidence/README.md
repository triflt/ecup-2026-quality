# 700: Qwen pair positive-evidence reranker

## Гипотеза

Предыдущий симметричный pair reranker исправил 3/4 FP, но создал 10 FN и не
исправил ни одного исходного FN. Новый однофакторный screen меняет только
обучающую цель и направление residual: модель оценивает, подтверждает ли
положительный donor положительность query, и может менять только `0 -> 1`.

Frozen Qwen3-VL embedding, candidate lists, solution140 fusion/threshold/prior,
folds, seed и BAD route не меняются. Первый запуск — дешёвая falsification на
уже принятом packet folds 0/3; это diagnostic, не ship gate. При успехе нужен
ровно один untouched grouped fold, а не full5.

Precedent: retrieval-augmented classification with a learned relevance/verifier
head; переносится сюда как asymmetric evidence verification для редкого класса,
где symmetric similarity transfer доказанно создавал false negatives.

## Frozen screen gate

- исправлен хотя бы один исходный flammable FN;
- создано не более одного нового FP;
- net corrections-regressions положительный;
- candidate лучше equal-capacity query-only;
- donor-label permutation не даёт положительного gain;
- изменения только `0 -> 1`, BAD byte-identical;
- без grid, threshold tuning, full5, package или Public.

## Результат

Terminal reject за 4.19 секунды на одной H100. Модель не исправила ни одного
из трёх исходных FN и создала один новый FP: Macro `0.956895 -> 0.953457`,
flammable F1 `0.948905 -> 0.942029`, corrections/regressions `0/1`.
Threshold, seed и aggregation не подбираются; untouched fold и package закрыты.
