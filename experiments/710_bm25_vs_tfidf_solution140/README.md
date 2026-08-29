# Experiment 710 — BM25 versus TF-IDF inside solution140

Connected-safe five-fold CPU comparison. The only changed factor is the text
component supplied to the frozen base and LoRA fusion: current TF-IDF rank versus
a preregistered top-30 BM25 reliability-weighted neighbour score. Qwen scores,
folds, fusion weights and production thresholds remain fixed.
