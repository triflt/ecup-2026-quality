# 631: production refit of experiment 603

Status: **production archive is ready for one Public ablation**.

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
