# Experiment 693 — qwen4-causal

Status: CPU-tested, launch blocked until the accepted `qwen27-all` outer-safe
teacher schema and hashes are supplied. No GPU job, upload, submission or
Public stage is authorized by this directory.

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
closed targets have an accepted schema/audit and are generated outer-safely by
teacher fold k trained only on outer-train; fold-k validation targets are
forbidden.

Precedent: Distilling Step-by-Step. Transfer mechanism: a compact model keeps
the calibrated hard-label objective while learning a closed, grounded causal
decomposition that separates the sold object from a mentioned fuel.

