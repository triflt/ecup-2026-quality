# Experiment 687 — Qwen3.5-4B gradient-conflict probe

Status: `TERMINAL_ACCEPT_PROBE_REJECT_PCGRAD`.

## Terminal result

The remote probe and its independent verifier completed successfully with all
80 preregistered measurements: 16 fixed effective batches at each of optimizer
steps 0, 170, 340, 510 and 680. Training consumed zero outer-validation rows or
labels, zero sealed rows and no Public information.

The frozen scientific decision is
`REJECT_PCGRAD_LOW_RETAINED_RANK_SIGNAL`. The measured conflict was not a usable
PCGrad route because the non-conflicting rank component retained after the
hard-primary projection was below the preregistered minimum. This is a
successful diagnostic and a negative model-selection result: neither PCGrad nor
the norm-cap fallback is opened. The next distillation hypothesis must change
the supervision interface rather than tune rank-loss weights, thresholds or
gradient-composition rules.

## Hypothesis

The additive rank term in experiment 686 can improve ordering while hurting the
hard-label decision boundary because its gradient conflicts with the hard-BCE
gradient.  This experiment measures that mechanism directly before spending a
paired quality screen on PCGrad/SCKD.

## One changed factor

No deployable model factor changes.  The exact fold-3 experiment-686 candidate
trajectory is replayed while diagnostics measure `g_hard` and `g_rank` on 16
fixed effective batches at optimizer steps `0, 170, 340, 510, 680`.

The diagnostic passes do not update model or optimizer state, run with dropout
disabled, and restore Python, NumPy, CPU Torch and CUDA RNG states.  The same
fixed batches are used at every checkpoint.  The inherited transport verifies
the checksum of the label-free outer-validation file and then deletes that
copy.  The probe itself consumes zero outer-validation rows and reads zero
outer-validation labels, predictions or quality metrics.

## Primary metrics

- cosine `cos(g_hard, g_rank)`;
- effective norm ratio `0.5 * ||g_rank|| / ||g_hard||`;
- conflict rate and Wilson interval;
- hard-descent cancellation;
- asymmetric-projection retention;
- the same metrics split by LoRA q/k/v/o projections.

## Frozen PCGrad gate

Open a later asymmetric-PCGrad screen only when all are true:

- complete finite measurements;
- pooled conflict rate at least `0.25`, Wilson lower bound at least `0.15`;
- conflict rate at least `0.15` on at least three of five checkpoints;
- median cancellation on conflicting batches at least `0.05`;
- median effective rank/hard norm ratio at least `0.10`;
- median projected rank-gradient retention on conflicting batches at least
  `0.50`.

Stop PCGrad when pooled conflict is below `0.10` or its Wilson upper bound is
below `0.20`.  Large norm ratio without conflict routes to magnitude control;
high conflict with low retention routes to structured distillation.

## Runtime

One H100.  A single fail-closed job performs inline technical preflight, the
exact train-only trajectory and all measurements.  Expected cost is about
`1.2–1.6x` one experiment-686 student training run.  It emits only compact
remote JSON artifacts and is never a submission model.
