# Experiment 624 — semantic-v3 route integration

Status: `blocked_by_dependency`.

This stage integrates **one** component only after experiment 621, 622, or 623
passes its frozen development gates. The reference-route weights remain those
of original route 603. Any new threshold must be calibrated donor-only; a
second-level router trained on historical OOF scores is forbidden.

If none of 621/622/623 passes, this stage records original route 603 as the
cycle winner and performs no speculative fusion. No metric or artifact is
claimed while the upstream decision is unresolved.

## Frozen integration evaluator

`evaluate.py` is the fail-closed evaluator for the only currently eligible
component, experiment 623. It first recomputes the complete experiment-623
five-fold acceptance result from checksum-verified runtimes and artifacts. It
then replaces only the original Qwen3.5 logits in route 603. The two visual
rank components and every route weight remain exactly unchanged.

For each target fold and category, both reference and candidate route
thresholds are fitted exclusively on the other four folds. There is no trained
second-level router and no second candidate variant. The full route is accepted
only if all frozen classification, component-bootstrap, category and false
negative gates pass. On acceptance, the evaluator emits a canonically signed
`exp624_route_recipe_v1` manifest consumed by experiment 625. On rejection it
emits no recipe manifest and preserves original route 603.
