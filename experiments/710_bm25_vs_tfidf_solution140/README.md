# Experiment 710 — BM25 versus TF-IDF inside solution140

Connected-safe five-fold CPU comparison. The only changed factor is the text
component supplied to the frozen base and LoRA fusion: current TF-IDF rank versus
a preregistered top-30 BM25 reliability-weighted neighbour score. Qwen scores,
folds, fusion weights and production thresholds remain fixed.

## Acceptance criterion

The candidate must improve connected-safe five-fold Macro F1 and preserve both
category guardrails under the frozen solution140 fusion. Public data is not used.

## Reproduction

Run `evaluate.py` with the competition CSV and the frozen four-head, Qwen3-VL
and Qwen3.5 OOF arrays. The evaluator checks row IDs, records SHA-256 digests,
and refuses to overwrite an existing output.

## Result and decision

Pending. No metrics are recorded and no refit, packaging or Public evaluation is
authorized until the connected-safe evaluation has completed.
