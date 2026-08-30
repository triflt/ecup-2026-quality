# 719: final evidence-driven distillation selection

This immutable selector separates three claims that are easy to conflate:

1. a teacher-guided objective beats a matched gold-only student;
2. a package runs within deployment constraints;
3. a candidate is proven better than production solution 140.

The fixed-component experiment-716 package may establish claims 1 and 2, but
cannot establish claim 3 because the exact solution-140 component OOF is absent.
It is retained as a deployable causal experiment, never silently promoted.

Experiment 718 is selected only if its standalone calibrated OOF directly beats
140, the primary and connected-family gates pass, and its one-H100 runtime is
accepted. Otherwise the frozen allow-listed solution-140 package remains the
production champion. Every decision and artifact is checksum-bound in
`results/final_selection.json`, which records both the SHA-256 and exact server
path of the selected ZIP.
