# 673: fail-closed verified OCR coverage

This CPU-only audit consumes the accepted row-level output of experiment 660. It never
repairs OCR and never exposes an unavailable image payload to a downstream model.

The experiment verifies every frozen SHA, the 49,456-image key space, row/index alignment,
the `OCR_AVAILABLE` payload subset, and the exact semantic-v3 system-140 replay. It reports
coverage separately for errors, correct controls, categories, folds, and singleton semantic
families. Broad policy-marker counts are retrieval diagnostics only and must not be used as
a classifier or as evidence that OCR is correct.

This experiment deliberately leaves critical-span recall and unsupported rate unset. Its
only possible positive outcome is authorization of a new blind visual audit under experiment
674. It cannot authorize model training, rules, thresholds, Public submission, or H2 itself.

## Result

All frozen input hashes and 49,456 image keys passed. The available payload contains exactly
41,691 images; the other 7,765 images remain fail-closed, with zero payload violations.

At least one verified image is available for 480/484 exact-140 errors and 349/352 singleton
errors. Coverage is high on every fold (97.75%–100%) and in both categories. However, only
227/484 errors have every image available, so absence of a phrase in the partial OCR cannot
be treated as negative evidence.

Verified OCR contains at least one text region absent verbatim from title/description for
470/484 errors. A deliberately broad diagnostic regex finds a novel policy-like marker in
109 errors, but these counts are neither relevance labels nor a rule. H2 remains unaccepted
until experiment 674 measures visual critical-span recall and unsupported text.
