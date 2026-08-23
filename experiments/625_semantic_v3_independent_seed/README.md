# Experiment 625 — independent-seed reproduction

Status: `blocked_by_dependency`.

After experiment 624 freezes its winning development recipe, this stage trains
one independent seed on all five semantic-v3 folds. The recipe, data selection,
route weights, thresholds, renderer, and post-processing may not change.

Acceptance requires the effect to retain its sign, win at least four of five
folds, and satisfy every category and safety gate inherited from the winning
component. Failure returns the cycle to original route 603. No result is
claimed before experiment 624 selects a recipe.

## Frozen reproduction contract

The independent seed is `31415`; the source seed is `42`. This is the only
allowed change. Architecture, multitask loss, physical-fold data, selector,
training schedule, route weights, thresholds, and exact-substring renderer are
the checksummed experiment-623 recipe.

`run_fold.py` fails closed unless the supplied experiment-624 recipe manifest:

- accepts experiment 623 as its sole winning component;
- binds every frozen experiment-623 source checksum;
- binds full threshold contract
  `c89899069890b514331243fb71b27fec1890749d3dc40032e4e1ce96db03810a`;
- declares semantic-v3, unchanged reference-route weights, and zero sealed use;
- carries a valid canonical manifest checksum.

Each of the five physical folds is a separate one-GPU run. The wrapper reuses
the exact label-isolated 623 runtime and runner, changes only its seed, and
rewrites the output contract as experiment 625 with full provenance. It does
not evaluate, tune, launch work, or claim acceptance.

```bash
python3 experiments/625_semantic_v3_independent_seed/run_fold.py --help
```
