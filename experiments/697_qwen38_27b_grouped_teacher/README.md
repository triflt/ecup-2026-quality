# 697: Qwen3.8-27B grouped-v1 offline teacher

This is an offline teacher screen against the exact grouped-text-v1 protocol used
to evaluate solution 140. The 27B model is never a submission dependency.

- source data: the canonical competition CSV supplied through `ECUP_DATA`;
- images: the first local image under `ECUP_IMAGES/<id>`;
- folds: `validation/grouped_text_v1/folds.csv`;
- objective: binary class-only LoRA;
- screen folds: 0 and 3, one four-H100 process per fold;
- early evidence: report folds 0 and 3 as soon as both finish; folds 1, 2 and 4
  are already queued on the same GPU lanes so the full run does not lose time;
- deployment: distil into a replacement Qwen3.5-4B LoRA for the existing 140
  inference pass. No additional model pass is allowed.

The original hard selector used by experiment 642 was not committed and is not
available on the server. To avoid silently reconstructing a different selector,
this experiment freezes a label-only deterministic sampler: balanced BAD,
fivefold positive oversampling for flammable, and a seeded negative sample.

After a teacher passes the five-fold quality gate,
`generate_teacher_targets.py` can score the exact outer-train occurrence stream
with the frozen adapter for that outer fold. These targets exclude the outer
validation fold, preserve duplicates and explicitly declare that they are
in-sample within outer-train. They are separate from holdout predictions and are
not generated before teacher acceptance.

`assemble_oof.py` verifies every completed fold contract, checksum, prediction
schema and exact fold ID coverage before writing the canonical
`teacher_oof.csv`. Raw binary logits are preserved for causal analysis, but
they are never compared directly across independently trained fold adapters.
Evaluation first applies a label-blind average-tie percentile rank within each
`(fold, category)` and then selects thresholds nested-style on the other four
folds. The resulting v2 report is the immutable input for component and
distillation analysis rather than an implicit collection of loose prediction
files.
Runtime examples:

```bash
ECUP_DATA=<data.csv> ECUP_IMAGES=<images-dir> \
python experiments/697_qwen38_27b_grouped_teacher/build_runtime.py \
  --fold 0 --output-dir .local/runtime-f0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python experiments/697_qwen38_27b_grouped_teacher/run_fold.py \
  --fold 0 --runtime-dir .local/runtime-f0 --output-dir .local/output-f0
python experiments/697_qwen38_27b_grouped_teacher/assemble_oof.py \
  --experiment-dir experiments/697_qwen38_27b_grouped_teacher \
  --output-dir .local/oof
```

The first fold-3 attempt demonstrated that microbatch 4 had effectively no
headroom on the longest examples (a 52 MiB allocation failed with 47 MiB free).
Queued and restarted folds therefore use microbatch 2 with accumulation 8. The
effective batch remains 16; data order, objective, context length, learning rate
and number of epochs are unchanged.

Starting with folds launched after commit `df92242`, `run_fold.py` writes an
atomic sibling checkpoint such as `.local/output-f1.resume.pt` every ten
optimizer updates. It contains the LoRA tensors, AdamW moments, scheduler,
CPU/CUDA RNG states and the exact next occurrence offset. A restart with the
same command validates the fold, runtime, data/order, batch and LoRA contract
before resuming. Do not delete the checkpoint after a failure; a successful
fold removes it only after publishing `output_contract.json`. Folds 0 and 3
were already running when resume support was installed and therefore cannot
create checkpoints retroactively.
