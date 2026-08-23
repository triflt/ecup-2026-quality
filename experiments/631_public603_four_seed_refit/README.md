# 631: production refit of experiment 603

Status: **two full-data seed refits are running**.

Experiment 603 found a predeclared equal-probability mean of four Qwen3.5
seeds. This package turns that development-only signal into a separately named
Public ablation. It does not rewrite experiment 603's rejected promotion gate
and does not claim a new champion.

The only production change relative to the original route is replacing the
single Qwen3.5 probability with the fixed arithmetic mean of seeds `42`,
`31415`, `271828`, and `161803`. Route weights, thresholds, Qwen3-VL, robust
base, and family priors remain unchanged from production experiment 190.

Seeds `42` and `31415` already have full-data adapters. Two new one-GPU jobs
fit the missing seeds with the exact parent recipe. No teacher, pseudo-label,
threshold search, weight search, or Public feedback is used.

The four sequential Qwen3.5 passes may exceed the organizer runtime. Therefore
the archive is deliverable only after an official-image smoke test proves both
Public and Private limits. A failed runtime check is a terminal rejection, not
permission to silently change the ensemble.

Precedent: fixed-seed averaging is a standard variance-reduction mechanism;
the relevant local evidence is experiment 602/603. The Public submission is
useful as an architecture-level ablation because experiment 230 confounded a
second seed with newly selected route weights, whereas 631 freezes the original
production route.
