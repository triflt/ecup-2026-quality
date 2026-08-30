# Removed experiment packages through ID 652

These experiment directories were removed from the review surface because they
were never run, were superseded before measurement, or had a terminal negative
result that is already preserved in `reports/experiment-log.csv`. Removing a
directory does not turn a rejected result into a missing result: the compact
evidence below is the canonical reason not to repeat the branch.

| ID | What was tested | Why the full package is not retained |
|---:|---|---|
| 240 | Family-balanced repetition of positive flammable families | The original output was not retained; the factor-isolated successor 241 contains the interpretable result |
| 250 | Family-diverse selection of flammable negatives | Superseded before training because the recipe mixed factors; the isolated test moved to experiment 410 |
| 320 | Label-blind regulatory BAD selector plus linear residual | Terminal negative: locked Macro delta `-0.000211`, only `2/5` fold wins, connected-safe delta `-0.000214`, and delta after priors `-0.000653` |
| 340 | Family-contrastive prototype score | Cancelled before any metric; no scientific result to reproduce |
| 450 | Transaction-scope text view for flammable products | Terminal negative: Macro delta `-0.006744`, `0/5` fold wins, one correction versus six regressions and four extra false negatives |
| 480 | Train-only blockwise Fisher adapter merge | Prepared but never launched; no adapter, prediction or metric was produced |
| 500 | Position-only transaction-scope augmentation | Failed the label-blind semantic gate before GPU: `62/73 = 84.93%` strict preservation versus required `98%` |
| 610 | Stricter successor to experiment 500 | Found only 91 independent eligible pairs versus the frozen minimum 100; no manual review or GPU training was run |
| 625 | Independent-seed reproduction after route integration | Skipped because experiment 624 selected the unchanged 603 fallback; no jobs were launched |
| 627 | Final recipe freeze stage | Skipped by the same upstream gate; no frozen candidate existed |
| 652 | Optional fast production OCR gate | Prepared but never run and superseded by the fail-closed partial OCR dataset in experiment 660 |

The one reusable artifact from experiment 500 is retained as
`validation/legacy_exclusions/exp500_semantic_audit.json`. Experiment 622 uses
its row IDs only to prevent previously inspected examples from entering a new
blind review.
