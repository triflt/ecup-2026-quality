# Experiment 674 analysis

## Decision

`NO_GO_REJECT_VERIFIED_OCR_FEATURE_SCREEN` after 80/120 blind rows.

The frozen gate required at least 90% critical-span recall, at least 95% scope preservation,
at most 1% unsupported critical text, and at least 50 visual-critical cases. The first 80
ratings contained 69 visual-critical cases, 52 complete OCR captures, 66 scope-preserving
captures, and no unsupported critical text.

With 40 rows left, the most optimistic possible recall was `(52 + 40) / (69 + 40) = 84.40%`.
The reviewer assigned to the remaining block was stopped, because no completion could reverse
the failed recall gate.

## Error slices

- Exact-140 errors: 28/33 complete captures (`84.85%`), scope 33/33.
- Correct controls: 24/36 complete captures (`66.67%`), scope 33/36.
- BAD: 29/41 complete captures (`70.73%`).
- Flammable: 23/28 complete captures (`82.14%`).
- Fold recall: `63.64%`, `87.50%`, `70.59%`, `86.67%`, `60.00%`.

The failure is therefore missing or incomplete decisive text, not unsupported OCR: the latter
was 0/80. The lower control recall and scope failures also make absence-based rules unsafe.
Adding all available OCR as ordinary text cannot satisfy the preregistered H2 mechanism.

## Consequence

Experiment 675 is not built. H3 online decisive crops was conditional on H2 and is not
launched. No model, threshold, fusion weight, rule, or Public submission is authorized by this
audit. The 660 corpus remains a valid fail-closed offline artifact for diagnosis only.
