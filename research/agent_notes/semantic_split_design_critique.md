# Independent critique: semantic-family split design

Date: 2026-08-22
Scope: read-only review of the proposed label-blind cross-category graph with
`exact full text`, `exact name`, `digit-masked name`, exact hashes of all gallery
images, a conservative perceptual-image edge, a roughly `1/7` sealed component
holdout, and five development folds.

## Decision

The direction is correct, but the proposed graph is **not ready to be sealed in
its literal form**. Three changes are mandatory first:

1. do not connect two products merely because one arbitrary auxiliary gallery
   image is equal or perceptually similar;
2. gate generic exact-name and digit-masked-name keys before unioning them;
3. do not use ordinary `StratifiedGroupKFold` to choose the sealed seventh or
   the five development folds; assign whole components with a deterministic
   multi-stratum balancing optimizer.

Without these changes the split can replace near-family leakage with a different
failure: giant or semantically heterogeneous components, a badly imbalanced rare
class, and a sealed score that is too noisy to arbitrate close candidates.

## Evidence from the current 12,971 training rows

I reproduced the text graph across both categories with the current conservative
normalizer (lowercase, `ё→е`, whitespace collapse; empty keys excluded). Counts
below are cumulative after adding each edge family:

| cumulative edges | components | rows in non-singletons | mixed-label components | largest component |
|---|---:|---:|---:|---:|
| exact normalized `name + description` | 9,481 | 5,268 | 55 | 58 |
| plus exact normalized name | 8,956 | 5,858 | 68 | 193 |
| plus digit-masked normalized name | 8,210 | 6,813 | 73 | 193 |

No exact text/name/masked-name key crossed the two categories. Cross-category
DSU is still the right invariant because image evidence can cross categories,
but it does not currently change the text-only graph.

The largest exact-name component is `БАД для иммунитета`: 193 rows, **102
different normalized descriptions**, and labels `190/3`. This is evidence that
exact name is sometimes a generic listing template rather than product identity.
Digit masking additionally makes a 47-row component from variants of
`Шнур силиконовый d - ... мм, L - ... м`. It also masks digits inside semantic
or model tokens such as `B12`, `D3`, `CoQ10` and `K-206` under the naive regex.
Those merges may be useful as a stress family, but they are too broad to assert
unconditionally as product identity.

After all three text edge types, the rare flammable positive class has only 126
components for 198 rows; the largest such component contains 14 positives.
Applying ordinary seven-way `StratifiedGroupKFold(random_state=20260822)` gave
flammable-positive counts **44, 25, 26, 49, 24, 13, 17** and fold sizes from
1,723 to 2,047 rows. The target is about 28 positives and 1,853 rows. Therefore
“one SGKF fold is the sealed seventh” does not meet the intended stratification.

The old `connected_family_guard_v2` is narrower in two ways: it builds DSU
separately within each category and uses only exact normalized full text plus
exact equality of the **first-image** fp16 embedding. It finds 8,690 components
and excludes 1,764 historical cross-fold rows. The new topology properly tries
to remove those two limitations, but all-gallery images require stronger edge
semantics than first-image equality.

The full-image run from experiment 200 successfully hashed all 49,456 images
with zero download failures. It selected 723 BAD overrides using exact hashes
with support at least four, while perceptual overrides varied strongly by fold
(from 0 to 289) and the final configuration disabled them. This is not a direct
component-size audit, but it proves that repeated auxiliary images are frequent
and that the existing perceptual key is not stable enough to become an
unconditional DSU edge. Before sealing, the builder must emit the complete
per-image degree/component audit; the current saved experiment-200 artifact
contains only the accepted label mapping, not all row-to-hash memberships.

## Edge policy that is safe enough to implement

### Text edges

- Keep exact normalized full text as a strong edge. Version the canonicalizer and
  explicitly define HTML, Unicode, punctuation and whitespace handling. Never
  union a missing/empty sentinel.
- Exact name alone should be a **candidate edge**, not an automatic edge. Accept
  it when at least one label-free corroborator holds: sufficiently similar
  descriptions, a shared product image, or a specific/non-generic title policy
  fixed before looking at errors. Quarantine high-degree keys for manual review.
- For digit masking, preserve alphanumeric semantic/model tokens (`B12`, `D3`,
  `CoQ10`, `K-206`) and mask standalone quantities/model-number spans only under
  a versioned rule. Require enough remaining alphabetic content and the same
  corroboration as exact-name edges.
- Do not use labels, model scores or error lists to accept individual edges.
  Labels may be shown only in the post-build audit and in fold balancing.

### Image edges

Exact-byte hash collision is not the practical risk; reused banners, ingredient
tables, instructions and marketplace placeholders are. A single shared auxiliary
image does not imply that two sold products are the same family.

Use the following hierarchy:

1. an exact first/product-cover image may be a direct edge, after quarantining
   high-degree and low-information placeholder hashes;
2. an exact auxiliary-image match becomes an edge only with a second independent
   match or text/name corroboration;
3. a perceptual match never becomes an edge by itself for an arbitrary auxiliary
   image: require first-image position, or two independent image matches, or text
   corroboration;
4. normalize EXIF orientation and record image position, dimensions, aspect ratio,
   entropy and hash degree. Filter blank/low-information images;
5. start conservatively with agreement of two independent hashes (for example
   DCT-pHash and dHash), compatible aspect ratio, and a fixed tiny Hamming radius.
   Freeze the threshold after a label-blind manual audit of matched and nearest
   rejected pairs.

Every edge should retain `edge_type`, hashed key, gallery positions, degree and
which corroboration admitted it. Report results both for each edge family alone
and cumulatively. Any component above 2% of the dataset (259 rows), any abrupt
component-size jump, and any image key shared by dozens of rows must fail closed
and be audited, not silently accepted or discarded.

## Split assignment

Build one global component graph, then assign components—not rows—against the four
joint strata:

- `БАД × 0`;
- `БАД × 1`;
- `Легковоспламеняющиеся × 0`;
- `Легковоспламеняющиеся × 1`.

Binary label alone is invalid because label semantics and prevalences differ by
category. Use a deterministic integer/greedy-plus-swap optimizer with a frozen
objective over total rows and these four counts. The builder should fail if it
cannot satisfy predeclared absolute tolerances. For flammable positives, use an
absolute-count tolerance (ideally target 28 with deviation no more than two in
the sealed set); relative-percent tolerances are misleading for such a small
class. Apply the same optimizer to the five development folds after removing the
sealed components.

The sealed `~1/7` will contain only about 28 flammable positives. One false
negative therefore changes recall by roughly 0.036. Treat its Macro F1 as a
one-time directional confirmation with component bootstrap uncertainty, not as
a precise ranking device for many nearly tied candidates.

## What “sealed” must mean operationally

The holdout is not independent merely because its IDs have a new fold number.
The following are required:

1. Freeze graph code, edge parameters, data checksum, component membership,
   assignment seed/objective and artifact checksums before the next candidate.
2. Exclude sealed components from **all** training and selection: adapter
   training, hard-example scoring, vectorizer/vocabulary fitting, neighbor and
   shingle indices, priors, prototypes, calibration, fusion and thresholds.
3. Retrain the complete 190 baseline recipe on the development complement. The
   existing full-data/old-fold adapters are not valid sealed baselines because
   they have already trained on the future sealed rows.
4. Generate any hard-mining score for a development outer fold using donor-only
   inner OOF models. Reusing the historical global OOF cache preserves the known
   selector leakage.
5. Pre-register exactly one candidate comparison, its checkpoint/archive hashes,
   fixed inference recipe and acceptance rule before opening the holdout.
6. Keep sealed labels/evaluation outside ordinary experiment outputs; expose only
   training IDs to training jobs. Log the single unlock time and result, then mark
   the holdout retired. If researchers inspect its row errors, it is development
   data from that point onward.
7. Assert zero component overlap for sealed↔development and for every dev
   train↔validation fold, plus donor/index membership checks for every learned
   downstream component.

Because `data.csv` already contains all labels, filesystem secrecy cannot make
the set cryptographically unknown to a determined local user. The meaningful
seal is procedural: separate evaluator, no routine row-level output, immutable
pre-registration, one unlock and retirement.

## Required artifacts before accepting the topology

- immutable `rows` table with row ID, component hash, dev/sealed assignment and
  no raw text/image bytes;
- edge audit with counts by edge type, key degree, image position and
  corroboration rule;
- component audit: size quantiles, largest components, mixed-category counts,
  joint-stratum vectors and incremental component mergers per edge family;
- split audit: per-part joint counts, maximum deviations, component-disjointness
  invariants and checksums;
- a label-blind manual audit of high-degree keys plus perceptual accepted/rejected
  boundary pairs;
- null tests showing deterministic reproduction and that permuting labels changes
  only assignment balancing, never graph topology;
- strict nested-training audit proving that sealed and outer-validation rows never
  enter selectors, priors, calibration or thresholds.

## Final recommendation

Proceed with a new immutable topology, but call the first build a **graph audit**,
not `sealed_v1`. Seal only after generic-name masking and auxiliary-image rules
pass the component/edge audit and the custom assignment meets rare-class counts.
Keep `grouped_text_v1` for historical comparability; use the new topology for a
freshly retrained 190 baseline and one predeclared challenger. This is the
smallest design that addresses near-family leakage, cross-category image reuse,
selector leakage and repeated human adaptation without manufacturing false
confidence from a noisy “sealed” fold.
