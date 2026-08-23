# Experiment 626: independent human-audit protocol

The builder is run only after experiment 624 has selected one recipe and
experiment 625 has reproduced it. Its inputs are the final development OOF
predictions, the original label-free card fields, development fold membership,
and explicit ID manifests from every earlier audit or review.

## Freeze procedure

1. Supply exactly 200 experiment-490 IDs and exactly 300 experiment-622 IDs.
2. Supply every other prior audit/review ID manifest with repeated
   `--prior-id-manifest` arguments.
3. The builder rejects any non-development membership and requires exact OOF
   coverage. It excludes all supplied IDs, retains one row per semantic
   component, ranks blindly by a fixed namespace, and freezes 300 rows.
4. The packet contains the exact source card, offsets, span or `NO_EVIDENCE`,
   closed concept, verdict, and rendered explanation. All human fields are
   empty. The private manifest freezes the sample hash, packet hash, ordered
   IDs, input hashes, and exclusion sources before anyone rates a row.
5. Keep both files under `.local`; neither product data nor ratings belongs in
   Git. Copy the CSV before review. Never edit the frozen original.

## Human review

The reviewer checks the concrete evidence, object of sale, negation,
composition/completeness, verdict consistency, and unsupported facts. Use `NA`
only for the three scoped dimensions that genuinely do not apply. Every other
rating is `0` or `1`. A failed row needs a short note. `strict_pass=1` is valid
only when all applicable detailed checks pass and both critical flags are zero.

This is a real independent human review. Model-generated or fabricated ratings
are forbidden. A structural validator is not a substitute.

## Decision

The evaluator accepts only a separate, fully filled copy matching the frozen
sample, contents, and order. GO requires at least 282/300 strict passes, zero
critical unsupported claims, and zero scope-or-negation failures. A completed
audit that misses any gate is `rejected_by_gate`, not “completed” or accepted.
The evaluator never reads sealed data.
