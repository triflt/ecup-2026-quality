# Experiment 706 final record and postmortem

## Disposition

**REJECTED — never promote to final.**

The strict-OOF parent-anchored submission scored `0.8217663209` public
macro-F1. The production experiment-140 reference scored `0.8923976821`, so the
observed public delta was `-0.0706313612`.

This record covers the complete Qwen3.8-27B to Qwen3.5-4B investigation,
including rejected scratch arms, strict five-fold teacher OOF generation,
parent-anchored student selection, full-data refit, packaging, and the public
failure. Public scores below were read from the competition UI.

## Provenance

- Teacher experiment: `697_qwen38_27b_grouped_teacher`
- Execution experiment: `706_qwen35_parent_anchored_distillation`
- Teacher model: `Qwen/Qwen3.8-27B` (local revision supplied through the runtime)
- Student/deployment model: `Qwen/Qwen3.5-4B`
- Runtime: the frozen environment recorded by the experiment contract

Teacher folds `f0..f4` produced strict OOF scores: each target row was scored by
a teacher adapter that had not trained on that row. The final target contract
covered all five folds.

## Teacher result

The early three-fold screen was optimistic:

- screen folds `f0/f2/f3` macro-F1: `0.9263637`
- BAD F1: `0.9443941`
- Flammable F1: `0.9083333`

The completed strict five-fold OOF result was worse than experiment 140:

- teacher nested macro-F1: `0.9082987330517893`
- experiment-140 nested macro-F1: `0.9118425205786493`
- delta versus experiment 140: `-0.0035437875268600205`
- teacher BAD nested F1: `0.9461818181818182`
- teacher Flammable nested F1: `0.8704156479217604`
- teacher Flammable fold-1 F1: `0.75`

The completed five-fold result should have been a hard stop before the full
student refit.

## Student objective and data

The final student was not direct KL/logit imitation. It was a conservative
teacher-guided ranking objective initialized from experiment 140:

- unique Flammable rows: `5502`
- positive rows: `198`
- negative rows: `5304`
- optimizer updates: `40`
- scheduled pairs: `5320`
- learning rate: `1e-5`
- parent-logit SmoothL1 weight: `1.0`
- hard gold BCE weight: `0.15`
- teacher within-gold-class pair-ranking weight: `0.10`

The numeric teacher score was consumed to order examples within each gold
class. It did not replace the gold verdict and was not directly regressed as a
soft target. The 27B teacher was absent at inference.

## Held-out selection

The student trained on folds `0..3` and was selected on gold fold 4, which the
student had not seen. Selection used the standalone Qwen3.5 score:

| Candidate | Average precision | ROC AUC | Best-F1 diagnostic | Eligible |
|---|---:|---:|---:|---|
| exp140 parent | 0.9903516822 | 0.9996495807 | 0.975 | reference |
| update 10 | 0.9873903031 | 0.9995529133 | 0.975 | no |
| update 20 | 0.9889487278 | 0.9996254139 | 0.975 | no |
| update 40 | 0.9926841128 | 0.9997341647 | 0.975 | yes |
| update 80 | 0.9888326516 | 0.9996254139 | 0.975 | no |

Update 40 improved average precision by `0.0023324306` and ROC AUC by
`0.0000845840`, but did not improve the diagnostic F1. The reported `0.975`
was the best F1 after threshold selection on the held-out data; it was not the
macro-F1 of the exact production fusion.

## Final package

The full refit used all `5502` unique Flammable rows and published:

- package: `exp706-strict-oof-final-submit.zip`
- package size: `55,645,038` bytes
- SHA256: `223b42bf9d21733d002a462566c8ca768c5b395d2d58764fa9a984705f2cb464`
- OCR/PaddleOCR: absent
- teacher at inference: false
- BAD adapter: byte-exact experiment-140 adapter
- Flammable Qwen3.5 adapter: full-refit distill adapter
- raw-logit distill alpha: `1.0`

The runtime ensemble retained experiment 140 for BAD. For Flammable its final
fusion was `0.15 * base + 0.10 * Qwen3-VL + 0.75 * Qwen3.5-distill`. Thus the
unproven distill adapter controlled 75% of the Flammable ensemble.

The short three-row runtime smoke passed. The 600-row local smoke was not a
quality evaluation; it only checked execution/schema/runtime and therefore
could not validate the competition metric.

## Public submissions

| Package | Public macro-F1 | Decision |
|---|---:|---|
| experiment 140 reference | 0.8923976821 | retain |
| exp705 gold control | 0.8251401937 | reject |
| exp705 hard-negative | 0.8005050505 | reject |
| exp705 hard-anchored rank | 0.7968485712 | reject |
| exp706 routed hard `a=0.15` | 0.8354026846 | reject |
| exp706 strict-OOF final | 0.8217663209 | reject |

No exp705 or exp706 package is eligible for final selection.

## Root cause

This was a validation and integration failure, not a simple token-direction or
prompt mismatch. Training and packaged inference both used `logit(1) -
logit(0)` and the same binary prompt contract.

The decisive failures were:

1. The three-fold teacher screen was treated as sufficient evidence before the
   full five-fold OOF result was available.
2. The full teacher OOF result was worse than experiment 140, but the student
   pipeline continued.
3. Checkpoint selection optimized standalone Qwen3.5 AP/AUC rather than the
   exact end-to-end competition macro-F1.
4. `best_f1_diagnostic` used a threshold selected on the holdout and was not the
   deployed experiment-140 fusion threshold.
5. The package replaced the complete Qwen3.5 Flammable channel (`alpha=1.0`),
   which carried 75% of the final Flammable score, without exact packaged OOF
   validation.
6. Runtime smoke validation checked execution and output schema, not model
   quality.

## Required gates for any successor

A successor must not be packaged or submitted unless all of these pass:

1. Complete five-fold strict teacher OOF before any full refit.
2. Reject the teacher transfer if full OOF macro-F1 does not beat the parent.
3. Verify an `alpha=0` package reproduces experiment 140 predictions exactly.
4. Evaluate the exact packaged fusion, thresholds, priors, prompts, and score
   direction on held-out/O​​OF data; standalone adapter AP is insufficient.
5. Require macro-F1 improvement over `0.9118425206` overall and robustness per
   fold, especially Flammable fold 1.
6. Do not use public submissions as the first end-to-end metric check.

Until those gates exist, experiment 140 is the only approved final package.
