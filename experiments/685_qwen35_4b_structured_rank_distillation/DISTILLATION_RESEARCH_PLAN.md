# Distillation research plan

## Decision

The first priority is not a teacher larger than 27B. The current Qwen3.6-27B
already yields a stable five-fold flammable signal; experiment 681 destroyed
that signal by copying saturated absolute logits. First test whether
magnitude-invariant pairwise ranking transfers to Qwen3.5-4B. Teacher scale,
structured evidence and feature KD are separate later factors.

This plan is a task-specific synthesis of our artifacts and the primary
literature below. No paper proves this exact binary multimodal LoRA recipe.

## Confirmed local evidence

- Current 27B offline route: flammable F1 `0.843931 -> 0.907514`, 5/5 folds,
  corrections/regressions `25/3`, FN `24 -> 13`.
- Raw-logit KD: mean AP `-0.02814` against matched hard-BCE, routed Macro
  `-0.02632`, flammable F1 `-0.05138`, FN `+6`, corrections/regressions `2/8`.
- Old teacher train AP `0.997--1.000`, median absolute logit `8.25--8.75`; at
  temperature 2 its soft labels average only `0.016--0.023` for negatives and
  `0.948--0.962` for positives.
- Gemma E4B screen improved mean flammable AP by `+0.04986` but failed the
  deployment gate because outer0 Macro tied and corrections/regressions were
  only `1.2`. AP and thresholded decisions must remain separate gates.

## Method selection

| Method | Fit to our failure | Decision |
|---|---|---|
| Rank-normalized pairwise KD | Removes teacher scale/saturation and directly targets rare-class ordering | First experiment |
| Listwise/RankDistil | Preserves top ranking but is harder to keep stable with microbatches | After pairwise only |
| Per-sample logit standardization | Two-class logits collapse to plus/minus a constant and lose margin | Do not use first |
| Decoupled KD | Its useful non-target distribution degenerates with one non-target class | Reject for binary head |
| Relational/contrastive KD | Can transfer inter-item geometry and new-family structure | Later one-factor lane |
| Hidden/visual feature KD | Requires layer mapping/projector and more storage/memory | Late lane |
| Response/rationale KD | Useful only after deterministic evidence verification | After rank mechanism |
| On-policy KD | Strong Qwen precedent but one verdict token has no meaningful trajectory | Only with structured generation |

## R0 — CPU target and pair audit

Input: immutable experiment-662 teacher scores; no Public or sealed rows.

Build one deterministic pair manifest per fold. Require:

- exact occurrence and semantic-family binding, finite coverage 100%;
- every positive represented; no item contributes more than 1% of total pair
  weight;
- K=8 negatives per positive: four teacher-hard and four hash-uniform;
- at least 30% of realized pairs classified as teacher-hard;
- rank-normalized `q` histogram, disagreement slices and manifest SHA recorded;
- duplicates/family components never cross a target-generation boundary;
- intentional duplicate and wrong-fold negative tests fail closed.

Stop before GPU if pair coverage is degenerate, more than 80% of `q` values sit
at clipping limits, or target provenance cannot be proved.

Cost: CPU only, under two hours.

## R1 / 685A — current 27B rank-KD

Changed factor: add the fixed pairwise rank term. Train a shadow paired-BCE
control from the same initialization with the same pair manifest, ordering,
batches and optimizer steps.

Run order:

1. 8-row control and candidate forward/backward/save/reload smoke.
2. Outer0 control + candidate on two 1xH100 jobs.
3. Open outer3 only if outer0 AP `>0`, Macro `>=0`, FN non-increasing and
   corrections/regressions `>=1.2`.
4. Outer3 control + candidate on two 1xH100 jobs.

Screen acceptance:

- AP delta positive on both folds, mean `>=+0.010`;
- routed Macro positive on both, mean `>=+0.003`;
- pooled flammable F1 delta `>=+0.010`;
- FN do not increase;
- corrections/regressions `>=1.5`;
- singleton-family net corrections positive;
- BAD byte-identical.

Cost: about `6--8 H100-hours`; teacher GPU cost 0. Public 0.

Immediate rejection: any sign failure, BAD drift, FN increase, or gain limited
to known repeats. No T/lambda/threshold/stop-step grid under 685A.

## R2 / 685B — nested teacher targets

Open only if R1 is positive-but-below-full gate, or R0 shows that in-sample
teacher ranking is too perfect to expose useful pairs.

For each student outer `k` and inner semantic-family fold `h`, train current
27B teacher `T(k,h)` on `outer-train(k) minus h` and score only `h`. Reuse the
exact R1 student objective and paired control; the only factor is target scope.

Screen cost: eight inner teachers for outer0/3, estimated `80--128 H100-hours`.
Full five-fold target set: about 20 teachers, `200--320 H100-hours`. Run at most
four 4xH100 teachers concurrently and preserve at least one submission lane.

Use the same R1 screen gates. R2 must beat R1 on mean AP by at least `+0.005`
to justify the nested cost.

## R3 — five-fold student confirmation

Run folds1/2/4 only after an R1 or R2 screen acceptance. Full acceptance:

- AP and Macro wins at least 4/5; folds1/2/4 Macro-positive;
- pooled AP delta `>=+0.010` and mean Macro delta `>=+0.006`;
- flammable F1 delta `>=+0.012`; FN do not increase;
- corrections/regressions `>=1.5`;
- semantic singleton delta positive;
- bootstrap `P(Macro gain > 0) >=0.90`;
- BAD byte-identical.

If AP improves while F1/FN fail, do not weaken the gate. Register a separate
nested calibration experiment whose threshold is selected exclusively within
outer-train.

## R4 / 684 — stronger Qwen teacher

Only after rank transfer is shown to work, compare teachers with the accepted
student loss fixed.

1. Qwen3.8-27B class-only LoRA versus Qwen3.6-27B: same data, prompt, objective,
   folds and LoRA recipe. Its previous training was infrastructure-blocked, not
   quality-rejected.
2. Qwen3.5-122B-A10B only if Qwen3.8 does not supply enough ceiling and a
   max-shape LoRA smoke proves memory/runtime. Model size is not a gate.

Teacher gate: AP positive on both screen folds, mean `>=+0.010`, no FN increase,
corrections/regressions `>=1.5`; full wins at least 4/5. Then substitute only
the teacher targets in the exact accepted student recipe.

Qwen3.8 screen estimate: `~32 H100-hours`. The 122B budget remains blocked
until topology and measured smoke exist; a larger capacity gap can hurt the
4B student even when teacher standalone quality rises.

## R5 — structured verified evidence

Open after rank-KD acceptance. Teacher emits only machine-checkable JSON:

```json
{
  "sold_object": "enum",
  "regulated_substance": "enum",
  "relation": "sold|included|compatible|mentioned|device",
  "evidence": [{"source": "text|ocr|image", "pointer": "span-or-bbox"}],
  "verdict": 0
}
```

Accept a target only if schema is exact, text evidence is extractive or image
bbox/crop exists, relation/verdict pass a frozen rule engine, repeated teacher
generations agree and training verdict matches the gold outcome. Free-form CoT
is discarded.

Manual audit gate: `>=282/300` full records correct, `>=95/100` sold-object and
relation correct, evidence present for `>=294/300`, unsupported `<=3/300`.

Student comparison changes only the auxiliary structured/evidence loss; rank
loss, sampler and all R1 parameters stay frozen. Use the same screen/full gates.

## R6 — independent Gemma same-family lane

The task-specific reason is Gemma E4B's mean AP gain `+0.04986`. Use a larger
multimodal Gemma teacher to produce the same rank-normalized targets, and train
Gemma E4B with the exact R1 rank loss. This is independent from Qwen and may
form the diverse second final candidate.

Official same-family Gemma distillation is mechanism precedent, not proof that
narrow LoRA KD will work. Teacher-only smoke and folds0/3 gates remain mandatory.

## R7 — feature/relation KD

Only after rank and structured lanes. Freeze one teacher layer/token view and
one projection into the student dimension. Compare feature loss against an
identical rank-KD control without changing rationale targets. Stop on memory,
runtime or singleton-family instability.

## Full refit and submission

Final targets must match the accepted CV target distribution. If nested targets
win, create a full all-row OOF teacher target set with semantic families kept
together; do not switch to in-sample full-teacher logits. Sealed target scoring
must be a separately bound label-free artifact.

Package only the 4B student adapter. Require SHA/self-hash, exact adapter/base
binding, null-route parity, BAD byte parity, 600-row mixed-category runtime
smoke, official format audit and projected 20/40-minute limits. Public is one
predeclared architectural check after the full local gate; no retuning from its
score.

## Primary literature

- Qwen3 Technical Report — official strong-to-weak off-policy/on-policy KD:
  https://arxiv.org/abs/2505.09388
- Qwen3-VL Technical Report — official same-family VLM distillation:
  https://arxiv.org/abs/2511.21631
- Gemma 2 Technical Report — official 27B to 9B/2B token-distribution KD:
  https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf
- Gemma 3 Technical Report — multimodal family trained with KD:
  https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf
- RankDistil, AISTATS 2021 — preserving teacher ranking:
  https://proceedings.mlr.press/v130/reddi21a.html
- Contrastive Partial Ranking Distillation, CVPR 2024 — relative ordering of
  hard negatives in image-text retrieval:
  https://openaccess.thecvf.com/content/CVPR2024/html/Chen_How_to_Make_Cross_Encoder_a_Good_Teacher_for_Efficient_CVPR_2024_paper.html
- Logit Standardization, CVPR 2024 — teacher/student scale mismatch:
  https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html
- Decoupled KD, CVPR 2022 — target/non-target decomposition:
  https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html
- CLIP-KD, CVPR 2024 — feature and contrastive VLM transfer:
  https://openaccess.thecvf.com/content/CVPR2024/html/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.html
- Visual Program Distillation, CVPR 2024 — execute/filter reasoning targets:
  https://openaccess.thecvf.com/content/CVPR2024/html/Hu_Visual_Program_Distillation_Distilling_Tools_and_Programmatic_Reasoning_into_Vision-Language_CVPR_2024_paper.html
- Distilling Step-by-Step, ACL 2023 — rationales as auxiliary supervision:
  https://aclanthology.org/2023.findings-acl.507/
- Teacher Assistant KD — excessive teacher/student capacity gap:
  https://arxiv.org/abs/1902.03393
