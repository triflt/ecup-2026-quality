# 674: blind verified-OCR critical-span audit

This audit is the missing H2 evidence gate. It freezes 120 unique semantic components:
30 exact-140 errors and 30 correct controls in each category. Three strata have exactly six
rows per fold. The flammable-error stratum uses every remaining independent error component,
distributed `8/8/8/5/1` over folds. Rows reviewed in experiment 490, the four actually rated
rows from experiment 670, and all generated experiment-672 pilot rows are excluded. Unrated
packets are not treated as observed evidence.

The reviewer sees category, product card, every source image URL, per-image availability,
and only the deduplicated regions from `OCR_AVAILABLE` images. Gold label, exact-140
prediction, error state, fold, and semantic component are private. `OCR_UNAVAILABLE`
images remain explicit and have no text payload.

The review records whether a visually decisive policy span exists, whether verified OCR
captures all critical text, whether its scope/relation is preserved, and whether OCR invents
critical text. A pass authorizes only experiment 675, a frozen OCR-consumer screen. It does
not authorize full training or Public submission.

The frozen packet contains 120/120 unique components and has SHA-256
`6607de6d12917f7e18423fdec7192172fae940bf0cb3c96576cac3e48252d043`. The private
manifest SHA-256 is `cb12490d10847bd732d52c937af8d95062dc93dd372ad27b7901576ef35948e6`.
The unfilled scorer returns `WAIT_FOR_COMPLETE_120_ROW_REVIEW`.

## Result

The audit stopped after 80 blinded rows because the recall gate became mathematically
impossible. Reviewers found 69 visual critical spans. Verified OCR captured all critical text
for 52 (`75.36%`) and preserved scope for 66 (`95.65%`). It introduced zero unsupported
critical claims in 80 rows.

Recall was `84.85%` on exact-140 errors and `66.67%` on correct controls. By category it was
`70.73%` for BAD and `82.14%` for flammable. Fold recall was `63.64/87.50/70.59/86.67/60.00%`.
Even if all remaining 40 rows were perfect critical captures, final recall could reach only
`84.40%`, below the frozen `90%` gate.

Decision: `NO_GO_REJECT_VERIFIED_OCR_FEATURE_SCREEN`. Experiment 675, full training, online
crop OCR, and Public submission are not authorized. This result rejects ordinary injection of
the partial OCR corpus; it does not erase the descriptive fact that OCR contains useful novel
text. A future OCR proposal needs a fundamentally different completeness mechanism and a new
non-overlapping audit, not threshold or lexicon tuning on these rows.
