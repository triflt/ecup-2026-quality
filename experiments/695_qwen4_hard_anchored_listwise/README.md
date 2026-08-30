# Experiment 695 — qwen4-rank

Status: CPU-tested against the experiment-692 selected winner (691/root or
696/fivefold), exact fold targets/report, routed acceptance and winner gate. Evidence is
not opened by the training method; the post-freeze evaluator opens only exp692-bound
evidence slices. One H100 job first runs one same-label pairwise fold0 smoke,
then candidate training and inline evaluation for all five folds sequentially.
Frozen 641 OOF predictions are the shared control.

Candidate sees all-category rows in the frozen parent order and keeps full hard
BCE coefficient 1.0. It uses the literal formula `L = L_hard + 0.5 * L_rank`
only among flammable rows inside the same hard-label stratum. There is no cap
and no lambda grid; boundary protection comes from never constructing a rank
pair across hard-label strata.
BAD production predictions are frozen baseline bytes.

This does not duplicate 685/686: those objectives rank opposite-label pairs;
695 never compares across the hard boundary and trains on the complete
all-category support. Precedent: learning-to-rank distillation with a supervised
anchor. Transfer mechanism: preserve classification calibration while learning
teacher ordering within positive and negative flammable strata.
The consumer binds target bytes to the selected teacher report, routed acceptance
and winner gate, and attaches `score` only to flammable
training occurrences.

The shared evaluator binds exp692 evidence slices, reports bootstrap and
singleton results, and gates the frozen
27/27 label-zero NAME policy slice against candidate FP regression.
