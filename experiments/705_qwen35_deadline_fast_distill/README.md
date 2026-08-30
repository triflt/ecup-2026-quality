# Experiment 705: deadline fast full-data distillation

This is an explicitly non-OOF fast track requested under the submission
deadline. It uses the completed fold-3 27B teacher adapter to score the full
training set, then trains three Qwen3.5-4B LoRA arms on all 6,790 sampled
occurrences in parallel.

The teacher-scoring runtime is derived from exactly those 6,790 occurrences
with source fold 3 removed. This guarantees that every consumed full-data
flammable ID is either scored in the outer train or supplied by the saved
fold-3 validation predictions; no sampler-coverage assumption is made.

Fold-3 validation IDs receive out-of-fold teacher scores. IDs from source folds
0, 1, 2, and 4 receive in-sample scores from the same fold-3 teacher. Contracts
record this limitation and cannot be consumed by the normal guarded exp715 path
without `--deadline-fast-track`.

The outputs are submission candidates, not causal proof of distillation. The
formal exp697/698/715–719 pipeline remains recoverable from checkpoints.
