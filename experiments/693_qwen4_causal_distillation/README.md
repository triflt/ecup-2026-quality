# Experiment 693 — qwen4-causal

Status: CPU-tested against both experiment-691 `foldK` and experiment-696
`fivefold/foldK` teacher layouts. Launch is blocked until experiment 692 emits
the exact `exp692_all_data_teacher_winner_v1` gate and its selected
`exp692_qwen27_routed_acceptance_v1`. The consumer requires both by file and
self SHA-256. No GPU job, upload,
submission or Public stage is authorized by this directory.

One H100 job first runs one real changed-factor fold0 F/B memory smoke, then
trains the candidate for all five semantic-family folds and evaluates inline.
The accepted frozen 641 production predictions are the shared hard-BCE control;
they are not retrained three times. The only candidate change is four physical
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
authoritative 27/27 label-zero NAME slice containing both `топлив` and `зажигал`,
reports baseline/control/candidate FP, and rejects any candidate FP increase.
The final scientific gate additionally requires at least 4/5 fold wins, pooled
Macro delta >=0.006, flammable F1 delta >=0.012, positive AP delta, FN
nonincrease, corrections/regressions >=1.5 (serialization-safe `inf` for zero
regressions), bootstrap probability >=0.90, strict singleton gain, BAD exact,
and no evidence slice with more regressions than corrections. Runtime and peak
GPU memory is forwarded from candidate output contracts when present. Without a
submission-minute limit the otherwise-passing result is resource-stage-pending.
