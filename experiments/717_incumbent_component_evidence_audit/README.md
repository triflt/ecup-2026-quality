# 717: incumbent component evidence audit

This experiment answers a narrow but necessary question before claiming that a
distilled Qwen3.5 adapter improves solution 140: are the exact per-row robust
base, Qwen3-VL and original Qwen3.5 OOF scores still available?

The audit is read-only with respect to QC and the main repository. It binds the
canonical experiment-140 metric, the later locked-190 manifest, the production
adapter hashes/configurations and the superficially similar QC step2 score
arrays. The latter are explicitly rejected as substitutes because they were
produced by all-linear adapters with different weight sizes and target modules.

If the three exact OOF files are absent, no exact fixed-fusion delta may be
claimed. Standalone grouped OOF, the connected-family guard and the isolated
adapter replacement package remain valid evidence, but fusion weights and
thresholds stay frozen.
