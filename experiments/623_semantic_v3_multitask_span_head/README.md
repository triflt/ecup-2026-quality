# Experiment 623 — semantic-v3 multitask span head

Status: `rejected_by_full_gate`.

The five-fold result improved Macro F1 from `0.8876054080` to `0.9033708111`
(`+0.0157654031`) and passed every category, safety, paired-change, and grouped
bootstrap gate. It nevertheless won only folds `0/1/3`, or `3/5`, below the
frozen requirement of `4/5`; folds `2/4` lost `0.0151667231/0.0125968118`.
The component is therefore retained for research and explanations but is not
integrated into the route, independently reproduced, or submitted.

## Hypothesis

The original experiment-600 Qwen3.5 verdict can improve when the same forward pass must
also point to a grounded phrase and classify that phrase into a small closed ontology.
The experiment does **not** generate chain-of-thought and does not use a teacher.

## Architecture

The parent verdict path is unchanged: the verdict logit is the difference between the
next-token logits for atomic tokens `1` and `0`. The last hidden-state sequence from that
same pass feeds four small heads:

1. start-token logits;
2. end-token logits;
3. a separate `NO_EVIDENCE` logit for each boundary distribution;
4. a five-way concept head.

The five renderer concepts are `OBJECT_OF_SALE`, `COMPOSITION`, `COMPLETENESS`,
`FUEL_OR_IGNITION`, and `NEGATION`. A prediction can only be rendered as an exact
substring of the canonical name/description text. Invalid boundaries or an unknown
concept fail closed to `NO_EVIDENCE`.

## Frozen objective

`CE_verdict + 0.10 * quality * (CE_start + CE_end) + 0.05 * quality * CE_concept`.

`quality=1` only for a SAFE exact-span candidate produced without looking at validation
labels. Rows lacking such evidence have all rationale terms masked; verdict CE remains.
Weights are frozen before GPU results and must not be adjusted afterward.

## Label isolation

`build_fold_runtime.py` physically writes two files. `train.jsonl` contains only the
four donor folds and may contain labels and rationale targets. `validation.jsonl`
contains the outer fold but has neither a label nor rationale field. Sealed rows are
discarded before either file is created. Semantic components crossing folds, mismatched
registries, duplicate IDs, incomplete evidence manifests, and non-empty output targets
all fail closed.

The evidence manifest is the development-only label-free manifest built by experiment
620. Only after the outer split is fixed does the builder use a donor label to choose
the matching candidate. Validation membership and validation model input never use this
operation.

## Run order

The CPU preflight and both frozen screen folds passed. Fold 0 improved Macro F1 by
`+0.082078`, fold 3 by `+0.015772`; the mean delta is `+0.048925`. The candidate
corrected 92 decisions and regressed 42, reduced flammable false negatives by 4,
and passed every predeclared classification and safety gate. Folds 1, 2, and 4
are now running without recipe changes. This directory intentionally contains no
task preset, data, weights, predictions, or sealed-holdout output.

## Runnable fold contract

`run_fold.py` imports the checksum-locked experiment-600 parent to reuse its data
selector, prompt, image preprocessing, LoRA configuration, batch size, accumulation,
schedule, and seed. It joins exact evidence to the unchanged parent prompt through
tokenizer offsets; missing, ambiguous, compacted-away, or otherwise unsafe alignments
are masked rather than repaired. The outer runtime has no labels or rationales.

Each successful fold writes a PEFT adapter ZIP, a separate safetensors auxiliary head,
label-free validation scores and grounded explanations, a selection audit, and a
hash-complete output contract. The runner rejects a changed parent, model revision,
runtime hash, nonempty output/image directory, failed image download, selector leakage,
non-finite score, or non-exact rendered evidence. Downloaded screen artifacts passed
these checks before labels were opened. Structural exact-substring coverage is `58.4%`
overall (`68.1%` BAD, `45.2%` flammable); this is not a human-quality score.
