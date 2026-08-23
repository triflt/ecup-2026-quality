# Experiment 628 — one-time sealed evaluation

Status: `blocked_by_dependency`.

The sealed semantic-v3 holdout may be opened exactly once, only after experiment
627 freezes one recipe that has passed the complete development and independent-
seed gates. Nothing may be tuned from this evaluation.

The evaluation must report Macro F1, both categories, family/component
bootstrap, singleton versus repeated items, false negatives,
corrected/regressed counts, and explanation quality. The sealed holdout has not
been opened by this stage and no metric is claimed.
