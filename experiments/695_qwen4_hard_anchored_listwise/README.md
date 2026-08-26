# Experiment 695 — qwen4-rank

Status: CPU-tested against the actual experiment-691 fold-specific
`teacher_targets.jsonl`, fold report, and authoritative exp692 routed acceptance. Evidence is
not opened by the training method; the post-freeze evaluator opens only exp692-bound
evidence slices. One H100 job runs paired control/candidate training and
inline evaluation for all five folds sequentially.

Both arms see the same all-category rows in the same order with the same model
initialization, seed, batches and steps. Both keep full hard BCE coefficient
1.0. Candidate uses the literal frozen formula `L = L_hard + 0.5 * L_rank`
only among flammable rows inside the same hard-label stratum. There is no cap
and no lambda grid; boundary protection comes from never constructing a rank
pair across hard-label strata.
BAD production predictions are frozen baseline bytes.

This does not duplicate 685/686: those objectives rank opposite-label pairs;
695 never compares across the hard boundary and trains on the complete
all-category support. Precedent: learning-to-rank distillation with a supervised
anchor. Transfer mechanism: preserve classification calibration while learning
teacher ordering within positive and negative flammable strata.
The consumer binds target bytes to `exp691_fold_report_v1`, binds each fold
to exact `exp692_qwen27_routed_acceptance_v1`, and attaches `score` only to flammable
training occurrences.

The shared evaluator binds exp692 evidence slices, reports bootstrap and
singleton results, and gates the frozen
25/25 label-zero NAME policy slice against candidate FP regression.
