# Experiment 625 — independent-seed reproduction

Status: `blocked_by_dependency`.

After experiment 624 freezes its winning development recipe, this stage trains
one independent seed on all five semantic-v3 folds. The recipe, data selection,
route weights, thresholds, renderer, and post-processing may not change.

Acceptance requires the effect to retain its sign, win at least four of five
folds, and satisfy every category and safety gate inherited from the winning
component. Failure returns the cycle to original route 603. No result is
claimed before experiment 624 selects a recipe.
