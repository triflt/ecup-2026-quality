# Experiment 693 — qwen4-causal

Status: CPU-tested against the actual experiment-691 `foldK/teacher_targets.jsonl`,
`evidence.jsonl`, `report.json`, and experiment-692 routed-acceptance schemas. Launch is
blocked until authoritative experiment 692 emits
`exp692_qwen27_routed_acceptance_v1` with `OPEN_THREE_STUDENT_METHODS`. The
consumer requires that acceptance by exact file SHA-256. No GPU job, upload,
submission or Public stage is authorized by this directory.

One H100 job runs paired hard-BCE control and candidate sequentially for all
five semantic-family folds and evaluates each fold inline. Both arms use the
same Qwen3.5-4B revision, seed, all-row hard BCE, source rows, shuffle, update
count and threshold. The only candidate change is an auxiliary causal target on
flammable rows: teacher hard verdict plus closed `sold_object`, `substance`,
`relation`, and evidence identifiers. Auxiliary output is training-only and is
never used at inference. BAD predictions in production replay are copied from
the frozen production baseline and must be byte-identical.

This is a student-stage successor to 689, not a repeat of 681/685/686: it uses
no soft logits and no cross-label rank loss. It may run only after the 689-style
closed targets are byte-bound by the fold report. The consumer uses only
grounded, non-abstaining flammable evidence and never merges validation evidence
into student training.

Precedent: Distilling Step-by-Step. Transfer mechanism: a compact model keeps
the calibrated hard-label objective while learning a closed, grounded causal
decomposition that separates the sold object from a mentioned fuel.
