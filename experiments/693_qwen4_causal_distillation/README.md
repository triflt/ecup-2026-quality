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
count and threshold. The only candidate change is four separate physical
auxiliary targets on flammable rows: `sold_object`, `substance`, `relation`, and
an evidence pointer. Each component has its own loss. Teacher verdict is never
an auxiliary target: the verdict target remains the dataset hard gold label in
the full all-row BCE. Auxiliary output is training-only and is never used at
inference. BAD predictions in production replay are copied from
the frozen production baseline and must be byte-identical.

This is a student-stage successor to 689, not a repeat of 681/685/686: it uses
no soft logits and no cross-label rank loss. It may run only after the 689-style
closed targets are byte-bound by the fold report. The consumer uses only
grounded, non-abstaining flammable evidence and never merges validation evidence
into student training.

Precedent: Distilling Step-by-Step. Transfer mechanism: a compact model keeps
the calibrated hard-label objective while learning a closed, grounded causal
decomposition that separates the sold object from a mentioned fuel.

The evaluator binds exp692 evidence bytes, reports its frozen evidence slices,
semantic singletons and paired bootstrap probability. It also verifies the
frozen 25/25 label-zero NAME slice containing both `топлив` and `зажигал`,
reports baseline/control/candidate FP, and rejects any candidate FP increase.
The final scientific gate additionally requires at least 4/5 fold wins, pooled
Macro delta >=0.006, flammable F1 delta >=0.012, positive AP delta, FN
nonincrease, corrections/regressions >=1.5 (serialization-safe `inf` for zero
regressions), bootstrap probability >=0.90, strict singleton gain, BAD exact,
and no evidence slice with more regressions than corrections. Runtime and peak
GPU memory are forwarded from paired output contracts when present. Without a
submission-minute limit the otherwise-passing result is resource-stage-pending.
