# Selected-teacher consumer compatibility

The consumer is bound to the experiment-692 winner gate and accepts either
experiment 691 at `foldK` or experiment 696 at `fivefold/foldK`. It opens only
the exact routed acceptance selected by that gate.

## P0 blockers found

1. A raw teacher fold report cannot select a teacher or open student training.
2. Experiment 692 performs routed evaluation for both teachers, selects the
   winner using the frozen head-to-head gate, and emits the selected acceptance.
3. All students require exact winner and acceptance file/self hashes, canonical
   AP/F1/FN gates, frozen BAD identity, and per-fold report/target/evidence
   bindings. Only 693 opens accepted evidence bytes; 694/695 consume accepted
   teacher targets only.

The consumer ignores BAD teacher scores, verifies exact ordered occurrence
identity for every target, requires `scope=all`, binds every candidate output
to the selected per-fold teacher artifacts, and never reads evidence for
methods 694/695.
