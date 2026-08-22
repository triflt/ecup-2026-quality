# 622: contrastive evidence verifier on semantic-v3

Status: **blocked before GPU**. The CPU builder and frozen contracts are ready;
the mandatory fresh human audit has not been performed. No score is claimed.

## Hypothesis

Experiment 603 showed that the four-seed route makes only 54 OOF changes. The
most useful error slices are asymmetric: direct BAD evidence gives 14
corrections versus 8 regressions, indirect BAD gives 4 versus 10, and long text
gives 6 versus 13. Experiment 622 tests whether a verifier trained on exact
evidence and deterministic counterclaims can reject unsupported route changes
without teaching a free-form chain of thought.

The classifier and final verdict remain frozen. The prospective verifier sees
an exact text span and one claim from a closed ontology. A positive donor target
uses the outer-train label; its negative keeps the same span and changes only
one predeclared semantic relation:

1. BAD identity assertion versus explicit negation;
2. included flammable item versus explicit exclusion;
3. sold flammable object versus compatibility, empty-container or integrated
   component scope.

There is no visual claim, inferred absence, external model or large teacher.

## Physical label isolation

The builder accepts three physically separate inputs for each outer fold:

- label-free features: `id`, category, name and description;
- label-free development membership and semantic components;
- donor labels containing exactly the IDs outside the outer validation fold.

It rejects a label-like column in either label-free input, any sealed row, any
missing or extra donor ID, any validation ID in donor labels, and any semantic
component crossing folds. Candidate extraction for both possible verdicts runs
before donor labels are applied. Validation-bank rows never contain a label.

## Frozen screen

Only folds 0 and 3 may be used for the reject-screen. The loss is frozen as:

- binary evidence-entailment CE: `1.00`;
- contrastive InfoNCE: `0.20`, temperature `0.07`;
- scope auxiliary CE: `0.15`;
- negation auxiliary CE: `0.15`.

The screen requires positive metric delta on each fold, mean delta at least
`+0.001`, no added flammable or safety false negatives, more corrections than
regressions and at least five changed decisions. Passing the screen cannot
accept the experiment; the full five-fold gates are in `frozen_spec.json`.

## Mandatory audit blocker

The builder deterministically draws 300 unique-family donor targets and excludes
every row ID found in legacy experiment 490/500 audit files. Raw review rows are
private and ignored. The public report contains aggregates only.

GPU work is prohibited until human review records at least `282/300` strict
passes, zero critical unsupported claims and zero scope/negation failures. Those
ratings do not exist yet, so the current decision is `PRE_GPU_BLOCKER`.

## CPU preflight

```bash
python experiments/622_semantic_v3_contrastive_evidence_verifier/build_preflight.py \
  --features label_free_features.csv \
  --membership label_free_development_membership.csv \
  --donor-labels-0 donor_labels_fold_0.csv \
  --donor-labels-3 donor_labels_fold_3.csv \
  --private-output-dir experiments/622_semantic_v3_contrastive_evidence_verifier/.local/preflight \
  --public-summary experiments/622_semantic_v3_contrastive_evidence_verifier/analysis/preflight_summary.json
```

The command does not train a model. It writes private candidate banks, donor
targets and an unreviewed audit sheet, then writes a sanitized aggregate summary.
