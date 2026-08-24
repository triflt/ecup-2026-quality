# Experiment 681 — deployable 4B flammable logit distillation

Status: `SCREEN_FOLDS_RUNNING`. GPU jobs: `2`. Public submissions: `0`.

## Defect and mechanism

The deployable Qwen3.5-4B class-only route misses part of the flammable ranking
signal that the offline Qwen3.6-27B probe recovered consistently on all five
frozen folds. The 27B model is too large for the competition runtime and is
strictly forbidden from the submission.

Experiment 681 transfers only the binary flammable logit into a fresh
Qwen3.5-4B LoRA. The 4B model, prompt, preprocessing, selector, image budget,
LoRA configuration, optimizer, learning rate, batching, seed, update count and
inference route remain frozen to the accepted 641 control. The only scientific
factor is the flammable training objective:

```text
hard = BCEWithLogits(student_logit, gold_label)
soft = BCEWithLogits(student_logit / 2, sigmoid(teacher_logit / 2)) * 4
loss = 0.5 * hard + 0.5 * soft
```

BAD predictions and its production adapter are not retrained or blended.

## Leakage-safe target contract

For student outer fold `k`, only teacher `k` may provide targets. Teacher `k`
was trained on the student outer-training rows and never saw outer fold `k`.
It scores those same outer-training occurrences. This keeps the student outer
validation blind, but the targets are in-sample and may be overconfident; this
risk is explicit and is the reason for the two-fold screen.

Ordinary five-fold OOF target merging is forbidden for outer CV because a
teacher for fold `j != k` has seen labels from student outer fold `k`.

Rowwise OOF scores are leakage-safe for a final all-data refit, but they are
not recipe-equivalent to the validated outer-CV targets: outer CV uses
fold-matched, in-sample teacher scores. The frozen target-shift diagnostic found
44--70 sign disagreements per 2,280 occurrences, logit MAE 1.36--1.91 and
2--5x more ambiguous OOF soft targets on folds 0/1/3. Therefore the existing
OOF full-runtime builder is fail-closed. It may run only after a separately
preregistered OOF-refit ablation policy is accepted. A recipe-equivalent final
refit instead requires a full-data 27B teacher which scores its own training
rows; neither route is authorized merely by a successful 681 CV gate.

The final 2,590-occurrence flammable multiset contains 2,242 development and
348 sealed-holdout occurrences (1,562/236 unique IDs). The five accepted 654
OOF files cover only development. Therefore full refit also requires a separate
label-free teacher-0 score artifact for the exact 236 selected sealed IDs, with
raw-logit semantics, labels read 0, Public 0 and exact immutable binding. This
artifact is forbidden before the full CV gate; missing sealed targets close the
refit rather than falling back to hard labels silently.

Every target artifact must bind exact `(global_index, id, fold, occurrence)`
keys, raw finite pre-sigmoid teacher logit difference, source adapter SHA,
source runtime SHA and payload SHA. Labels and Public data are forbidden in all
teacher-score artifacts. Sealed rows are forbidden in outer-CV artifacts; the
post-gate final-refit completion artifact is the only explicit exception and is
scored label-free.

## Frozen launch order

1. Accept technical teacher-scoring smoke and all required target manifests.
2. Build and verify fold 0/3 student runtimes; run one 8-row technical smoke.
3. Train folds 0 and 3 on at most two H100 GPUs in parallel.
4. Evaluate against exact 641 control predictions without changing thresholds,
   weights, rules or stop step.
5. Open folds 1/2/4 only if the screen gate passes.
6. After the full gate, choose and preregister a final-target policy. Full-data
   refit remains closed until either a recipe-equivalent full-data in-sample
   teacher is available or the OOF-refit distribution shift is explicitly
   accepted as a separate ablation.
7. Package only a refit that passes the target-policy gate and the full runtime
   rehearsal, or the separately preregistered Public-ablation fallback below.

## Metrics and gates

Tie-aware Average Precision is exactly equivalent to
`sklearn.metrics.average_precision_score`. It is the primary rare-class ranking
diagnostic. Final decisions also require Macro F1, flammable F1, FP/FN,
corrections/regressions, semantic singleton families and exact BAD parity.

### Screen gate (folds 0 and 3)

- AP delta positive on both folds and mean AP delta at least `+0.005`.
- production Macro delta positive on both folds and mean at least `+0.0015`.
- pooled flammable F1 delta non-negative; flammable FN do not increase.
- corrections/regressions at least `1.5`.
- BAD predictions byte-identical.

Any failed condition is terminal `REJECT_AT_SCREEN`; no tuning of `T`, lambda,
threshold, weights, rules or step is allowed under experiment 681.

### Full primary gate

- AP delta positive on at least 4/5 folds; mean AP delta at least `+0.005`.
- Macro wins at least 4/5; folds 1/2/4 all positive; mean Macro delta at least
  `+0.006`.
- pooled flammable F1 delta at least `+0.012`; FN do not increase.
- corrections/regressions at least `1.5`.
- bootstrap `P(Macro gain > 0) >= 0.90`.
- singleton-family Macro delta positive; BAD byte-identical.

### Preregistered Public-ablation fallback

One Public submission may be built if the primary gate fails only by magnitude,
while all of the following hold: AP positive on at least 4/5 folds with mean
delta at least `+0.003`; Macro wins at least 4/5 with folds 1/2/4 positive and
mean delta at least `+0.004`; pooled flammable F1 delta at least `+0.008`; FN do
not increase; corrections/regressions at least `1.5`; bootstrap probability at
least `0.85`; singleton delta positive; BAD byte-identical. A sign failure,
fold-instability failure, FN regression, BAD drift or leakage closes this
fallback.

## Deployment contract

The submission contains only the already available competition base models and
4B/2B deployable components. It must contain no 27B base, adapter or runtime.
The existing all-row Qwen3.5-4B pass is retained byte-for-byte so BAD is not
affected by category-dependent batching. Flammable rows then receive one
additional pass with the distillation adapter and the exact 641 image-area
preprocessing. Before upload: null-route parity, mixed-category official smoke,
archive manifest/SHA, format audit and a full runtime rehearsal must pass.
