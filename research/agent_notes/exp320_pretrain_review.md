# Independent pre-training review of experiment 320

Date: 2026-08-22. Scope: read-only review before any residual-head fit.

## Blocking findings corrected before training

- Repeat topologies no longer inherit historical label-derived thresholds. Historical
  comparability keeps the frozen nested thresholds; all repeats use the single
  production threshold `0.27193570137023926`.
- The raw first-image 2048-dimensional embedding was removed because its extraction
  path differs from the all-images production path in experiment 190. The residual
  now has 35 reproducible evidence/locked-score features and needs no new VLM pass.
- Selector, locked archives, Qwen score archives and connected rows are bound by
  SHA-256. IDs, folds, labels, categories, locked score and uncertainty fail fast on
  mismatch.
- Mixed-label connected components remain in outer evaluation but cannot supply
  training evidence.
- Medicine, food and veterinary cohorts now have explicit correction/regression and
  aggregate F1 guardrails.
- Experiment 320 can start only from a typed `post310_decision.json` bound to at
  least three completed experiment-310 audits. A manual environment flag cannot
  bypass the branch.
- A fail-closed gate combines historical, connected-safe, repeat-topology,
  deterministic order-stability and downstream-prior results. Offline acceptance
  only authorizes full refit; it is not production promotion.

## Validation version

`component_transfer_gate_v3` preserves all numerical v2 requirements and adds an
explicit deterministic-head stability protocol: seed-31415 donor-order permutation
on the best and worst folds with identical predictions required. Stochastic models
continue to use the v2 seed repeat.

## Deliberately unresolved until an offline pass

`build_submission.py` remains blocked. Only if the complete offline and prior gates
pass may the project add a full-data serialized Ridge component, hidden-schema
feature-parity test, organizer-runtime smoke with reserve and package integrity
audit.
