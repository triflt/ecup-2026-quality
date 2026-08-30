# Experiment 707 — supervised CLIP pair reranker

Frozen `openai/clip-vit-base-patch32` representations replace the Qwen embedding
inside the already accepted balanced hard-pair screen. Two simultaneous arms are
preregistered: multimodal CLIP and text-only CLIP. Candidate lists, folds,
solution140 decisions/priors, pair rows, seed, optimizer, aggregation and gates
remain frozen. BAD never changes.

The multimodal representation is the concatenation of normalized text, image,
elementwise-product and absolute-difference vectors (2048 dimensions). Missing
images receive a zero image vector. The text-only control pads the same normalized
text vector to 2048 dimensions.

PASS requires at least one original flammable FP and one FN corrected, positive
net corrections, at most one regression, no FN increase, multimodal improvement
over text-only and query-only, and no positive donor-permutation gain. No cutoff
or weight tuning after metrics.
