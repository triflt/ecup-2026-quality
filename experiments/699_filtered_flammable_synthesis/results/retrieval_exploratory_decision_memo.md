# Exp699 exploratory submission decision

Selected candidate: `solution140_flammable_graph_aware_soft_cache_v1`.

This is deliberately an exploratory package, not a validated champion. It adds
one orthogonal, training-free signal to the frozen solution140 flammable route:
a reliability-gated soft cache over 4,716 labeled flammable donors. BAD, both
Qwen adapters, base models, fusion weights, thresholds, and the annotator prior
remain byte-identical. The cache acts after the existing ensemble rank score and
before the existing prior.

The manifest-bound cross-fitted OOF result is negative: 28 final decisions
changed, with 5 corrections and 23 regressions. It removes five flammable false
positives but creates 23 false negatives, moving Macro F1 by `-0.03031134` and
flammable F1 from `0.95652` to `0.89590`. This is the central shipping risk and
must not be hidden. Packaging nevertheless proceeds because the exploratory
lane was explicitly required even when local validation is weaker, provided
the candidate is deterministic, provenance-valid, runtime-valid, and nonzero.

Alternatives were rejected for the exploratory slot because they are less
identifiable: matched-order synth10 tied the equal-dose real control on final
decisions, TF-IDF synth had 0/5 routed wins, targeted relabel lacks untouched
confirmation, and conflict weighting was negative on the exact package route.

The final ZIP is stored only in remote compute at
`/workspace/runs/exp699/soft_cache_package_v2/solution140_flammable_soft_cache_v1.zip`.
Its SHA-256 is
`98a0e02e8cde8c04fde529461ce4b813a352a15fefacec289f3d5e8843c89eae`
and its size is 56,094,581 bytes. Two independent builds are byte-identical;
the ZIP has 276 safe members and passes integrity testing. The extracted ZIP
passes the existing eight-row end-to-end schema smoke, with BAD output exactly
equal to solution140. Status is `ExploratoryReadyNotSubmitted`; ODS submission
still requires explicit action-time confirmation.
