# Experiment 685 — rank-first distillation into Qwen3.5-4B LoRA

Status: `P0_HARDENED_CPU_VENDOR_BRIDGE_PENDING`. GPU jobs: `0`. Public: `0`.

The first experiment transfers the already proven Qwen3.6-27B flammable
ranking into a deployable Qwen3.5-4B LoRA. A larger teacher is not required for
685A: experiment 681 failed because absolute in-sample logits were saturated,
not because the current teacher lacked outer-fold signal.

BAD remains byte-identical. The large teacher is offline only and never enters
the submission.

## 685A — cheapest valid mechanism test

Reuse the accepted fold-matched teacher scores from experiment 662. Teacher
`k` trained only on student outer-`k` train and scored those same training
rows. This is outer-validation safe because fold `k` labels never enter teacher
or student training, but it is explicitly in-sample and may have weak hard-pair
diversity.

Convert teacher logits to empirical normal ranks within each target block:

```text
r_i = clip(Phi^-1((rank(t_i) - 0.5) / n), -2.5, 2.5)
q_ij = clip(sigmoid((r_i - r_j) / 0.5), 0.05, 0.95)
```

For each positive item, deterministically select eight negatives from the same
block: four closest by teacher rank and four hash-seeded uniform negatives.
Normalize loss per positive.

```text
L_hard = mean_pairs 0.5 * [BCEWithLogits(s_i, 1) + BCEWithLogits(s_j, 0)]
L_rank = mean_pairs BCEWithLogits(s_i - s_j, q_ij)
L_candidate = 0.5 * L_hard + 0.5 * L_rank
```

A shadow control uses the exact same pair manifest, batches, steps and model
recipe but `L_control = L_hard`. Candidate versus shadow control therefore
changes only the ranking term. Absolute raw-logit BCE is forbidden.

Frozen student recipe: Qwen3.5-4B, LoRA rank 16/alpha 32/dropout 0.05 on
q/k/v/o, seed 42, LR `2e-4`, one epoch, effective batch 16 and threshold 0.
No hyperparameter is selected on outer folds.

Run paired control/candidate on outer0 first. Outer3 opens only if outer0 has
AP delta `>0`, Macro delta `>=0`, no FN increase and corrections/regressions
`>=1.2`. The complete folds0/3 gate is mean AP `>=+0.010`, Macro positive on
both with mean `>=+0.003`, flammable F1 delta `>=+0.010`, no FN increase,
corrections/regressions `>=1.5`, positive singleton net and BAD byte-identical.

The outer0 decision has a prebuilt CPU-only remote fast path. It consumes the
two accepted S3 training outputs, binds fold0 validation rows to labels held in
the accepted fold3 outer-train runtime by exact
`global_index/id/fold/category`, and emits a self-hashed gate report. This is
not a shortcut around validation: the training jobs still read zero validation
labels, BAD is unchanged by construction, and the fast report is authorized
only to open fold3. Screen/full acceptance still requires the complete frozen
replay evaluator.

To reduce wall-clock time, the already preregistered fold3 control/candidate
pair may be trained speculatively after both fold0 jobs have started, provided
its code, runtimes, presets and output keys are frozen before any outer0
quality result exists. Speculative execution does **not** open fold3: only job
state and transport failures may be observed. Predictions, acceptance payloads
and quality metrics remain sealed in S3 until outer0 returns
`OPEN_SCREEN_FOLD3`; an outer0 rejection permanently freezes the unread fold3
outputs as unused. Folds1/2/4 are not eligible for speculative execution.

## 685B — nested targets only if warranted

Open 685B only if 685A is positive-but-limited or the CPU audit proves that
train-perfect ranks remove useful hard-pair diversity. Within student outer
fold `k`, split outer-train by semantic family into inner folds `h`; teacher
`T_{k,h}` trains without `h` and scores only `h`. Family duplicates never
split. The student loss and all constants stay identical to 685A.

Ordinary global five-fold OOF teacher merging is forbidden because another
outer teacher may carry labels from the current student validation fold.

Estimated nested-teacher cost is `80--128 H100-hours` for outer0/3 and
`200--320 H100-hours` for all five outer folds. This cost is not authorized
until 685A supplies evidence.

## Later one-factor extensions

- Experiment 684 substitutes a better Qwen teacher into the exact accepted
  rank-KD recipe.
- Structured causal/evidence supervision is separate: sold object, regulated
  substance, relation and extractive evidence pointer. Free-form chain of
  thought is never treated as truth.
- Feature/relational KD and Gemma-to-Gemma are independent later lanes.

Full program and literature mapping:
[DISTILLATION_RESEARCH_PLAN.md](DISTILLATION_RESEARCH_PLAN.md).

All preparation, training, evaluation and terminal-report artifacts remain in
remote compute/S3; the workstation holds only code, remote references/SHA and concise
summaries.
See [REMOTE_EXECUTION_CONTRACT.md](REMOTE_EXECUTION_CONTRACT.md). Local archive
workarounds are forbidden, and a submission ZIP is downloaded only after all
scientific/package/runtime gates pass and the user gives fresh approval.
Team ownership, messaging and GPU rules are frozen in
[TEAM_OPERATING_PROMPT.md](TEAM_OPERATING_PROMPT.md).

Image delivery is not changed in 685A: frozen runtime rows keep the proven 641
`image_url`, while `/work/images` is an ephemeral per-job cache. Introducing a
new persistent image cache would confound the rank-loss ablation.

The only legacy-input exception is a one-time, label-free remote compute bridge for
accepted 662 folds0/3. It verifies the canonical inner archive and score SHA,
then writes to an immutable S3 prefix. No legacy output is downloaded locally,
and all later stages read the S3 copy.

Training additionally consumes the proven PEFT 0.20.0 ZIP through a one-time
label-free CPU bridge. The bridge binds the historical source bundle SHA,
checks required modules and package metadata, imports the required PEFT symbols
from that exact ZIP, and publishes the ZIP plus a self-hashed acceptance to a
new immutable prefix. Code, pair input, base-model tree, initial LoRA state,
training order, runtime versions and vendor bytes are all carried into the
control/candidate acceptance and checked for exact parity.

The only approved object namespace is
`s3://biglm-alignment-pipeline/d.strizhakov/ecup/`. Direct S3 credentials may be
read only from the ignored `.env.s3` when generating ignored remote compute presets;
their values are never printed or committed.
