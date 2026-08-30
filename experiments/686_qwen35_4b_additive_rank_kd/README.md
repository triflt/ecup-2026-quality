# Experiment 686 — additive pairwise rank distillation

Status: `TERMINAL_REJECT_CONFIRMATION`.

## Terminal result

The blind fold-3 screen passed, but the preregistered folds 1/2/4 confirmation
failed. The additive rank term retained a useful ranking signal: mean flammable
AP increased by `+0.014763712`. That gain did not transfer stably to the frozen
decision boundary. Direct Macro-F1 improved by only `+0.003490358` pooled and
`+0.002687758` as the mean fold delta; fold 2 was negative, flammable F1 rose by
only `+0.006980716`, false negatives increased from 16 to 17, and the
correction/regression ratio was `10/8 = 1.25`.

The frozen production comparison was weaker: Macro-F1 changed by
`+0.000665090`, flammable F1 by `+0.001330180`, and false negatives increased
from 15 to 17. BAD predictions remained byte-identical. The terminal decision
is therefore `REJECT_CONFIRMATION`; fold 0, full replay, refit, packaging and
Public submission are closed. No lambda or threshold sweep is permitted. The
result supports the next preregistered mechanism: preserve the teacher ranking
signal while controlling rank-gradient conflict or dominance.

## Motivation

Experiment 685 transferred ranking information from the accepted 27B teacher into
the deployable Qwen3.5-4B LoRA on outer fold 0: flammable AP increased by
`+0.018345844`. It nevertheless failed the frozen deployment gate because
flammable F1 decreased by `-0.018945212`, false negatives increased from 8 to 9,
and corrections/regressions were 1/2. The 685 candidate used
`0.5 * hard_BCE + 0.5 * pair_rank_BCE`, so the rank factor was confounded with a
50% reduction of the proven hard-label objective.

Experiment 686 changes exactly one scientific factor relative to its paired
hard-BCE control: the candidate uses

`hard_BCE + 0.5 * pair_rank_BCE`.

The control uses `hard_BCE`. Both arms retain the same Qwen3.5-4B base revision,
LoRA, fold, rows, pair order, initialization, seed, learning rate, batch sizes,
optimizer steps, teacher pair packet, source runtime and threshold 0.

This additive formula deliberately has a larger total loss/gradient scale than
the control. Therefore 686 is a clean objective ablation, not a normalized
gradient-scale ablation. Hard, rank and total losses plus pre-clip gradient norms
are logged separately; no normalization is introduced after preregistration.

## Honest validation

- Outer fold 0 is development-only because it motivated this loss correction.
- The first scientific result is blind outer fold 3.
- Fold-3 promotion requires a frozen full-production replay with AP strictly
  positive, direct Macro/flammable-F1 delta strictly positive,
  flammable false negatives not increasing, corrections/regressions at least
  1.2, production Macro non-negative, and BAD byte-identical.
- The compact fold-3 evaluator is technical-only; it verifies remote bindings
  and full label-packet coverage but cannot issue a scientific promotion.
- Only `OPEN_CONFIRMATION_FOLDS124` permits folds 1, 2 and 4.
- Confirmation additionally requires AP positive on all folds 1/2/4, mean AP
  delta at least `+0.010`, direct Macro positive on all three with mean at least
  `+0.003`, flammable F1 delta at least `+0.010`, correction ratio at least 1.5,
  positive singleton-family net, non-increasing FN and non-negative frozen-route
  production Macro.
- Final acceptance requires positive direct Macro delta on all four blind folds
  1/2/3/4, mean four-blind-fold Macro delta at least `+0.006`, blind-fold
  flammable F1 delta
  at least `+0.012`, no increase in flammable false negatives, correction ratio
  at least 1.5, positive singleton-family net corrections and component
  bootstrap `P(delta>0) >= 0.90`. It also requires AP positive on all four blind
  folds with mean blind-fold AP delta at least `+0.010`, production Macro delta at
  least `+0.006` and production flammable F1 delta at least `+0.012`.
- Fold 0 may be replayed for descriptive five-fold reporting, but none of its
  metrics participates in promotion, final gates or hyperparameter selection.
- No threshold, weight or rule is selected on validation or Public.

## Leakage and transport contract

- The training bundle is physically label-packet-free and excludes
  `validation/semantic_family_v3/folds.csv` plus frozen replay artifacts.
- The evaluation bundle is separate and is the only bundle allowed to contain
  the frozen semantic-v3 registry.
- Training reads zero outer-validation labels, zero sealed rows and zero Public
  information.
- Pair runtimes, transport acceptance and PEFT vendor bridge are reused from
  accepted experiment-685 inputs and remain bound by their exact SHA-256 values.
- S3/remote compute artifacts remain remote. Local ZIP downloads are forbidden unless
  a qualified submission package is being built with explicit approval.
- AppleDouble, unsafe archive members, provenance mismatch or access-control
  errors fail closed. Corporate DLP/EDR/quarantine controls are never bypassed.
- Promotion is not authorized by the evaluator JSON alone. The independent main
  integrator must produce a strict `promotion_gate.json` bound to a terminal
  `SUCCEEDED` evaluator job, immutable S3 output source, job-metadata SHA, raw
  evaluation SHA, code/replay/registry SHA, exact prediction/acceptance SHA and
  an all-true frozen gate map. Training mounts and verifies both the gate and the
  original evaluation. A caller-made minimal self-hashed JSON is rejected.

## Immediate stop conditions

- Any control/candidate difference besides the frozen rank-loss term.
- Any training-bundle inclusion of the semantic registry or replay labels.
- Fold 3 has non-positive AP, negative direct Macro delta, more false negatives,
  or corrections/regressions below 1.2.
- Any validation-label, sealed-row or Public access by training.
