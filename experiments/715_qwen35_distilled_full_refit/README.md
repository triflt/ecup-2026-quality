# 715: promoted Qwen3.5-4B full-data refit

This stage is fail-closed. It runs only when experiment 698 promotes a
distillation candidate through both the primary five-fold gate and
`connected_family_guard_v2`. If neither candidate passes, no full adapter is
trained and the gold control remains the evidence-backed fallback.

The full runtime scales the frozen label-only recipe in the same way as the
historical Qwen3.5 full refit: up to 1,900 examples per BAD label, every
flammable positive repeated five times, and up to 2,000 flammable negatives.
It retains the exact canonical 12,971-row data and immutable grouped fold IDs.

Teacher targets are constructed without another 27B training run. Every ID uses
its strict OOF score from the one experiment-697 teacher that excluded that ID.
Hard-negative weighting consumes the raw binary logit. Ranking consumes a
category-and-source-fold percentile so scores from five separately calibrated
teacher adapters are never compared on an invalid raw scale.

The selected method is the promoted candidate with the larger primary nested
Macro delta. Selection accepts only the v2 evaluation/guard contract in which
every fold-model score is calibrated by label-blind `(outer fold, category)`
percentile rank; raw logits from separately trained adapters are never compared
across folds. A changed-factor smoke must succeed before the single-H100 full
refit. The output contains only a Qwen3.5-4B rsLoRA adapter and cryptographic
bindings to the teacher OOF, student evaluation, connected guard, full runtime
and selected method. The 27B teacher is never an inference dependency.

The refit imports the same versioned first-image `448x448` Pillow-thumbnail
preprocessing as every experiment-698 arm and records that version in its
output contract. Packaging rejects an adapter whose preprocessing contract does
not match the unchanged solution-140 Qwen3.5 runtime pass.
