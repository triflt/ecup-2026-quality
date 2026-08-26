# Experiment 694 — qwen4-hardneg

Status: CPU-tested against the actual experiment-691 fold-specific
`teacher_targets.jsonl`, fold report, and authoritative exp692 routed acceptance. Evidence is
not opened by this method. One H100 job runs all five folds sequentially, each with a
plain hard-BCE control followed by the candidate and inline evaluation.

Both arms use every BAD and flammable training occurrence exactly once, full
hard BCE coefficient 1.0, the same initialization, seed and optimizer-step
count. The sole scientific difference is order: the candidate presents
outer-safe teacher-identified flammable hard negatives and missed positives
first; the control retains the frozen seed-42 order. There are no soft logits
or teacher probabilities in the loss. BAD production predictions are replayed
byte-for-byte from the frozen baseline.

This is not experiment 681 (no soft-label loss) or 685/686 (no pairwise/rank
loss). Precedent: hard-example mining and curriculum learning. Transfer
mechanism: spend early, high-learning-rate updates on flammable boundary errors
identified by a stronger teacher without changing the supervised objective.
The consumer binds target bytes to `exp691_fold_report_v1`, binds each fold
to exact `exp692_qwen27_routed_acceptance_v1`, and attaches `score` only to flammable
training occurrences.
