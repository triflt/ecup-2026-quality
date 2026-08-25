# Experiment 685 — structured/ranking distillation into Qwen3.5-4B LoRA

Status: `BLOCKED_PENDING_ACCEPTED_684_TEACHER`. GPU jobs: `0`. Public: `0`.

## Objective

Transfer the flammable ranking and rule-understanding signal of an accepted
experiment-684 teacher into a deployable `Qwen/Qwen3.5-4B` LoRA. BAD stays
byte-identical to the production BAD route. The student is used only on
flammable rows and the large teacher is absent at inference.

Experiment 681 permanently rejects naive raw-logit KD. This experiment must
not reopen its temperature/lambda grid. The new mechanism is grounded
distillation of information that the raw saturated logits did not contain:

- cross-fitted teacher rank/margin;
- `sold_object`;
- `regulated_substance`;
- relation of the substance to the sold object (`sold`, `included`,
  `compatible`, `mentioned_only`, `container_or_device`);
- evidence span/image reference;
- final teacher verdict and confidence.

The exact schema is frozen after a 300-component teacher/evidence audit and
before any student GPU training.

## Why this differs from experiment 681

The old teacher scored its own outer-training rows. Its median absolute logit
was `8.25--8.75`; at temperature 2 the average soft target was approximately
`0.016--0.023` for negatives and `0.948--0.962` for positives. Teacher train
AP was `0.997--1.000`. Thus the soft term was nearly another hard label and
carried little dark knowledge about difficult ordering or sold-object scope.

New targets are nested cross-fitted inside each student outer-train split and
contain explicit decision structure. No teacher used for student outer `k`
may train on outer `k` labels. Outer validation remains unread until the
student artifact is frozen.

## Student control and changed factor

- backbone/revision: exact deployable Qwen3.5-4B;
- same flammable selector, prompt/image view, LoRA rank 16, alpha 32, dropout
  0.05, q/k/v/o targets, seed 42, one epoch, effective batch 16 and LR `2e-4`;
- same threshold and production routing as the hard-BCE control;
- only changed factor: hard-BCE objective becomes the preregistered
  structured/ranking distillation objective.

The loss weights are not selected on outer folds. Before GPU, one fixed loss
is chosen on teacher-only/inner-development evidence:

```text
L = L_hard_verdict
    + lambda_rank * L_pairwise_teacher_rank
    + lambda_attr * L_causal_attributes
    + lambda_evidence * L_grounded_evidence
```

Absolute raw-logit BCE is not included by default. It may be a separate later
ablation only if teacher logits are demonstrably non-saturated and calibrated
on leakage-safe inner data. Pairwise/listwise ranking is primary because the
competition author explicitly identified PR-AUC as the appropriate diagnostic
for the rare class.

## Cheapest gates

1. Accepted 684 teacher beats the current teacher qualification gate.
2. Manual audit: at least `282/300` correct verdict+evidence records,
   `>=95/100` correct sold-object/relation cases, unsupported evidence
   `<=3/300`.
3. Target audit: exact occurrence binding, labels/Public/sealed violations 0,
   non-saturated rank coverage and positive teacher AP on all screen folds.
4. One 8-row forward/backward/save/reload technical smoke.
5. Student folds 0/3 only. Folds 1/2/4 remain closed until screen acceptance.

## Student screen gate

Against both the hard-BCE specialist and the production flammable route:

- flammable AP delta positive on folds 0 and 3, mean at least `+0.010`;
- routed Macro delta positive on both, mean at least `+0.003`;
- flammable F1 non-negative on both and pooled delta at least `+0.008`;
- FN do not increase;
- corrections/regressions at least `1.5`;
- singleton-family net corrections positive;
- BAD byte-identical.

Any sign failure is terminal for the frozen objective. No outer-fold tuning of
loss weights, temperature, threshold, route weight or stop step is allowed.

## Full acceptance and deployment

Full CV requires AP/Macro wins on at least 4/5 folds, folds 1/2/4 Macro
positive, mean Macro delta at least `+0.006`, pooled flammable F1 delta at
least `+0.012`, no FN increase, corrections/regressions at least `1.5`,
positive singleton delta and bootstrap `P(gain > 0) >= 0.90`.

Only after full acceptance may a full-data target policy be registered. The
submission then packages only the 4B adapter, passes null-route parity,
mixed-category 600-row runtime smoke and the 20/40-minute limits. The teacher
base/adapter and target artifacts are never packaged.

