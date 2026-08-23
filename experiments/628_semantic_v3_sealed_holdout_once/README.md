# Experiment 628 — one-time sealed evaluation

Status: `skipped_by_gate`.

Experiment 627 produced no accepted frozen recipe. The sealed holdout was not
opened, the reveal ledger remains absent, and the evaluation count remains zero.

The sealed semantic-v3 holdout may be opened exactly once, only after experiment
627 freezes one recipe that has passed the complete development and independent-
seed gates. Nothing may be tuned from this evaluation.

The evaluation must report Macro F1, both categories, family/component
bootstrap, singleton versus repeated items, false negatives,
corrected/regressed counts, and explanation quality. The sealed holdout has not
been opened by this stage and no metric is claimed.

`reveal_ledger.py` creates the reveal record with an atomic create-only write.
It binds the frozen recipe, acceptance policy, evaluator and sealed-input
identity and refuses a second record. Failure falls back explicitly to Public
champion 400 and local safe route 603 without tuning.
