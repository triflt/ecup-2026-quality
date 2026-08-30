# Experiment 622: blind human-audit protocol

The private packet contains 300 deterministically sampled rows from distinct
semantic product families. It contains no gold verdict and no model score. The
reviewer must not look up either while rating the packet.

For every row, read the product name, the complete source field and the quoted
exact surface span. Compare `claim_concept` with `counterclaim_concept`, then
fill the four initially empty review fields:

- `strict_pass`: `1` only when the quoted span, in its product context, directly
  supports `claim_concept`, rejects the paired counterclaim, preserves
  polarity and refers to the sold product or explicitly included item;
- `critical_unsupported`: `1` when the claimed fact is not stated by the span
  and context, requires outside knowledge, or reverses the expressed meaning;
- `scope_or_negation_failure`: `1` when exclusion, compatibility, emptiness,
  integrated-component scope, modality or negation is interpreted incorrectly;
- `review_notes`: a short factual reason for every failure; it may be empty for
  a strict pass.

Use only `0` or `1` in the three rating columns. A strict pass must have both
failure flags set to `0`. Any failure flag forces `strict_pass=0`. Do not change
any other column or reorder rows.

The GPU gate opens only at at least 282 strict passes out of 300, zero critical
unsupported claims and zero scope/negation failures. Until all 300 rows have
human ratings, the experiment remains blocked before GPU.
