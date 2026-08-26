# Experiment 691 consumer compatibility

The consumer is bound to the actual experiment-691 output layout:
`foldK/teacher_targets.jsonl`, `foldK/evidence.jsonl`, `foldK/report.json`, and
`fivefold_report.json`.

## P0 blockers found

1. Raw `exp691_fold_report_v1` does not bind evidence bytes, and raw
   `exp691_fivefold_report_v1` does not provide the canonical routed promotion
   decision. The running frozen teacher is not changed or retried.
2. Frozen experiment 692 performs the post-terminal routed evaluation and emits
   `exp692_qwen27_routed_acceptance_v1`. All three students require its exact
   file SHA, `OPEN_THREE_STUDENT_METHODS`, canonical AP/F1/FN gates, frozen BAD
   identity, and per-fold report/target/evidence bindings. Only 693 opens the
   accepted evidence bytes; 694/695 consume accepted teacher-target bytes only.

The consumer additionally ignores BAD teacher scores, verifies exact ordered
occurrence identity for every target, requires `scope=all`, binds every fold
report to the self-hashed fivefold report, and never reads evidence for methods
694/695.
