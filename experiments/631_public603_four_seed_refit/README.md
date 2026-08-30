# 631: production refit of experiment 603

Status: **Public ablation completed and rejected online**.

Experiment 603 found a predeclared equal-probability mean of four Qwen3.5
seeds. This package turns that development-only signal into a separately named
Public ablation. It does not rewrite experiment 603's rejected promotion gate
and does not claim a new champion.

The only production change relative to the original route is replacing the
single Qwen3.5 probability with the fixed arithmetic mean of seeds `42`,
`31415`, `271828`, and `161803`. Route weights, thresholds, Qwen3-VL, robust
base, and family priors remain unchanged from production experiment 190.

Seeds `42` and `31415` reuse verified full-data adapters. The two new one-GPU
fits for seeds `271828` and `161803` completed with the exact parent recipe;
their archives, reports, contracts and checksums passed. No teacher,
pseudo-label, threshold search, weight search, or Public feedback was used.

The first sequential implementation preserved the recipe but projected 44.93
minutes on Private. Reusing image loading and tokenization across the four
adapters reduced the official-image 600-row smoke to 331.95 seconds, or 14.75
minutes Public and 35.04 minutes Private. Its output is byte-identical to the
unoptimized implementation, the schema is valid, and all four adapters are
bound by SHA-256 in `results/production_manifest.json`.

Precedent: fixed-seed averaging is a standard variance-reduction mechanism;
the relevant local evidence is experiment 602/603. The Public submission is
useful as an architecture-level ablation because experiment 230 confounded a
second seed with newly selected route weights, whereas 631 freezes the original
production route.

The frozen archive received Public Macro F1 **0.8636091486**, which is
`-0.0287885335` below experiment 140. It is also slightly below the rejected
two-seed experiment 230. This causally cleaner result shows that the
development gain of fixed seed averaging does not transfer to the Public
distribution. No weights, thresholds or follow-up recipe are tuned from this
observation; the four-seed production route is rejected.
