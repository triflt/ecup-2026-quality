# 718: conditional distilled Qwen3.5 standalone

This experiment is the simplified deployment branch. It does not assume that
replacing the Qwen3.5 adapter inside solution 140 improves the full fusion. It
authorizes a one-model Qwen3.5 pipeline only when the selected distilled arm:

- passed the matched gold-control science gate and connected-family guard;
- exceeds the historical solution-140 nested Macro F1 by at least `+0.001`;
- wins at least four outer folds versus its matched gold control;
- does not trail either solution-140 category F1 by more than `0.005`.

Scores are label-blind percentile ranks within each `(outer fold, category)`.
Deployment uses the median of the five outer-train donor thresholds, never a
threshold optimized on the held-out predictions it scores. The full-data
adapter, evaluation, guard, preprocessing and thresholds are checksum-bound in
an immutable selection contract. A rejected selection produces evidence but no
standalone package.

The runtime contains only `run.py`, the selected Qwen3.5 adapter, a self-hashed
threshold config and metadata. It does not load OCR, text/embedding classifiers,
Qwen3-VL, empirical priors or the 27B teacher. This keeps architecture and OOF
causally aligned and should be substantially faster than solution 140. Package
construction remains conditional on the immutable authorization contract and a
real one-H100 smoke is still required before deployment acceptance.

The conditional runtime waiter first verifies the exact `mlitvinov_vlm` Torch,
Transformers, PEFT, Pandas and Pillow stack on one visible H100. If selection is
authorized, it runs a three-row schema smoke and a 600-row timing/VRAM smoke,
then creates an immutable acceptance report. Unlike solution 140, neither the
preflight nor runtime imports Paddle/OCR or mounts any model except Qwen3.5-4B.
It shares an advisory GPU0 lock with exp716 and holds it across preflight and
both timed smokes, preventing overlapping measurements or artificial OOMs.
