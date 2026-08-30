# Experiment 682 — BAD-only seed-632 route

Status: `development_full_runtime_ready_not_trained`.

## Confirmed five-fold result

The one changed factor is the Qwen3.5 logit source for category `БАД`:
seed-632 replaces the original score.  For `Легковоспламеняющиеся`, the raw
Qwen3.5 logits, probabilities, rank scores, final scores and predictions remain
byte-identical.  Robust-base scores, Qwen3-VL scores, route weights,
thresholds and donor-only priors are frozen to the semantic-v3 experiment-140
replay.

The materialized five-fold replay improves Macro F1 from `0.9104830823` to
`0.9133075471` (`+0.0028244648`), wins all five folds, changes BAD F1 from
`0.9539141414` to `0.9595630711` (`+0.0056489297`), corrects 72 rows and
regresses 19.  All-positive false negatives fall from 256 to 221; flammable
false negatives and all flammable metrics remain unchanged.  Semantic-component
bootstrap gives `P(delta > 0) = 1.0`, 95% interval
`[0.0014630942, 0.0044809387]` (10,000 resamples, seed 63531415).

These are validation facts, not a deployable-model claim.  No Public feedback
or sealed rows were used.

## Development-only full refit contract

The post-validation policy permits exactly one seed-31415 fit on the 11,118 CV
development rows, with zero sealed rows/labels and no Public feedback.  This is
now source-provable: the experiment-600 parent already contained its
`FULL_TRAIN=1` branch before this analysis (BAD cap 1,900 per class,
flammable-negative cap 2,000, flammable positives repeated five times).

The CPU builder does not accept competition data or the sealed registry.  It
reconstructs development supervision from five checksum-bound experiment-623
outer-train runtimes; every development row must occur with a byte-identical
payload in exactly four runtimes.  The exact frozen selector produces 6,116
occurrences / 5,436 unique rows, multiset SHA
`f4734df62119ca67d86e22681fb09544b8c56761c6ffaf7617875cccdb983e9f`,
1,529 microbatches and 383 optimizer updates.  Counts are BAD 1,633/1,633 and
flammable negative/positive-occurrence 2,000/850.  The old 12,971-row
`four_head_oof` is explicitly forbidden because it is another selector
distribution.

`train_full.py` implements the same verdict BCE + span/concept loss, optimizer,
schedule and LoRA recipe as 632 with `FULL_TRAIN=1`.  The launch gate is `GO_GPU`,
but no GPU job has been launched.  Packaging and Public remain closed until the
adapter and direct 600-row runtime smoke exist.

## Production design after the refit

`build_submission.py` reuses the exact champion-140 archive.  It retains the
original Qwen3.5 adapter, scores every row with it, loads the full seed-632
adapter, and overwrites only BAD positions.  It never loads or packages the
span/concept auxiliary head for the verdict.  A direct 600-row smoke must prove
schema validity, byte-identical flammable original scores, prediction parity
and the frozen 20/40-minute Public/Private ceilings before packaging can open.

The remote compute preset must be emitted below `.local/`, audited with
`scripts/audit_preset.py`, and dry-run with an explicit `--preset-file` before a
single live submission.  Current files authorize only that one frozen training
job; they do not authorize packaging or a Public submission.
