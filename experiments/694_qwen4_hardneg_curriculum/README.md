# Experiment 694 — qwen4-hardneg

Status: CPU-tested against the actual experiment-691 fold-specific
`teacher_targets.jsonl`, fold report, and authoritative exp692 routed acceptance. Evidence is
not opened by the training method; the post-freeze evaluator opens only exp692-bound
evidence slices. One H100 job runs all five folds sequentially, each with a
plain hard-BCE control followed by the candidate and inline evaluation.

Both arms use every BAD and flammable training occurrence exactly once, full
hard BCE coefficient 1.0, the same initialization, seed, teacher-guided order,
batches and optimizer-step count. The sole scientific difference is a frozen
1.0–2.0 hard-gold BCE weight for flammable examples where teacher verdict and
gold label disagree, scaled by bounded teacher confidence. There are no soft
targets replacing the hard gold label. BAD production predictions are replayed
byte-for-byte from the frozen baseline.

This is not experiment 681 (no soft-label loss) or 685/686 (no pairwise/rank
loss). Precedent: hard-example mining and curriculum learning. Transfer
mechanism: upweight flammable boundary errors identified by a stronger teacher
without changing target labels or data support.
The consumer binds target bytes to `exp691_fold_report_v1`, binds each fold
to exact `exp692_qwen27_routed_acceptance_v1`, and attaches `score` only to flammable
training occurrences.

The shared evaluator binds exp692 evidence slices, reports bootstrap and
singleton results, and gates the frozen 25/25 label-zero NAME policy slice
against candidate FP regression.
