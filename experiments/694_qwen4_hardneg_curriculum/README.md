# Experiment 694 — qwen4-hardneg

Status: CPU-tested against the experiment-692 selected winner (691/root or
696/fivefold), exact fold targets/report, routed acceptance and winner gate. Evidence is
not opened by the training method; the post-freeze evaluator opens only exp692-bound
evidence slices. One H100 job first runs one changed-factor fold0 smoke, then
runs the candidate over all five folds with inline evaluation. Frozen 641 OOF
predictions are the shared control; no duplicate control is trained.

The candidate uses every BAD and flammable occurrence exactly once, full hard
BCE coefficient 1.0 and the frozen parent order/batches/steps. The sole change is a frozen
1.0–2.0 hard-gold BCE weight for flammable examples where teacher verdict and
gold label disagree, scaled by bounded teacher confidence. There are no soft
targets replacing the hard gold label. BAD production predictions are replayed
byte-for-byte from the frozen baseline.

This is not experiment 681 (no soft-label loss) or 685/686 (no pairwise/rank
loss). Precedent: hard-example mining and curriculum learning. Transfer
mechanism: upweight flammable boundary errors identified by a stronger teacher
without changing target labels or data support.
The consumer binds target bytes to the selected teacher report, exact routed
acceptance and winner gate, and attaches `score` only to flammable
training occurrences.

The shared evaluator binds exp692 evidence slices, reports bootstrap and
singleton results, and gates the frozen 27/27 label-zero NAME policy slice
against candidate FP regression.
