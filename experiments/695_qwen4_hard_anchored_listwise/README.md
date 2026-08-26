# Experiment 695 — qwen4-rank

Status: CPU-tested, launch blocked on the immutable `qwen27-all` outer-safe
teacher artifact. One H100 job runs paired control/candidate training and
inline evaluation for all five folds sequentially.

Both arms see the same all-category rows in the same order with the same model
initialization, seed, batches and steps. Both keep full hard BCE coefficient
1.0. Candidate adds a fixed 0.20 listwise auxiliary only among flammable rows
inside the same hard-label stratum. The auxiliary is capped at 25% of detached
hard BCE, so it cannot trade away the hard boundary. There is no lambda grid.
BAD production predictions are frozen baseline bytes.

This does not duplicate 685/686: those objectives rank opposite-label pairs;
695 never compares across the hard boundary and trains on the complete
all-category support. Precedent: learning-to-rank distillation with a supervised
anchor. Transfer mechanism: preserve classification calibration while learning
teacher ordering within positive and negative flammable strata.

