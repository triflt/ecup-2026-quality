# Experiment 689 — grounded transaction-graph KD

Status: `PREREGISTERED_REMOTE_TARGET_AUDIT_ONLY`. Student GPU: `NO_GO`.
Jobs, uploads, presets and bundles: `0`.

## Hypothesis and parent lane

The hypothesis is that an auxiliary transaction graph can transfer the
distinction between what is sold, what substance is present, and how an item is
related to that substance more safely than free-form rationale distillation.
This is the structured-evidence lane preregistered in experiment 685. It is not
a continuation of terminally rejected experiment 688, and no student training
parent is selected by this packet.

The mechanism has precedent in ERASER's extractive-rationale evaluation and
CLARITY's evidence-to-concept structure. Distilling Step-by-Step motivates
auxiliary supervision, but experiment 689 deliberately removes its free-form
rationale channel: the teacher can return only closed graph enums and indices
of deterministic evidence candidates already present in its request.

No classification label, verdict, logit, probability, rank, family identifier
or free chain-of-thought is exposed to or accepted from the teacher. This stage
tests target quality only. It cannot authorize a model, GPU job or submission.

## Frozen target ontology

- `sold_object`: `device`, `fuel_consumable`, `accessory`, `kit`, `other`,
  `unknown`;
- `substance`: `gas`, `flammable_liquid`, `solid_fuel`, `ignition_aid`, `none`,
  `unknown`;
- `relation`: `primary_sold_object`, `included`, `compatible_external`,
  `mentioned_only`, `negated`, `absent`, `unknown`.

A teacher response has no free-text field. It contains only the request binding,
the three enums, sorted unique `evidence_candidate_indices`, `support_status`
and `supervise`. A `supported` response must select at least one existing
candidate and sets `supervise=true`. `unsupported` and `ambiguous` must select
no candidates and set `supervise=false`. The materializer never accepts or
creates an evidence span or an ontology code outside the frozen schema.

The exact schema is frozen in
`target_audit_schema_v1.json`; thresholds, enums, selection salt and teacher
surface are additionally bound by `frozen_target_audit_spec.json`.

## Remote-only source and disjoint sample

The scripts read no workstation competition file and contain no S3 bucket,
account, endpoint, credential or personal path. A future CPU-only remote compute job
must receive every input as a native mounted artifact below a caller-provided
`--remote-root`; every output must remain below that same root. S3 object choice,
remote compute input names and output publication are external runtime configuration.

The remote source JSONL contains opaque row/component/family tokens, a fixed
stratum, source text with hashes, and deterministic exact-span candidates. Its
self-hashed contract must state that labels and score-like fields are absent and
that sealed and Public row counts are zero. Separate self-hashed exclusion
manifests for experiments 670 and 672 contain only sorted SHA-256 row, component
and family tokens. Any overlap in any of those three namespaces is excluded.
The approved runtime sources must be supplied with their exact expected SHA-256;
the builder additionally requires 300 unique experiment-670 components, 40
unique experiment-672 components, zero intersection and union size 340. Their
locations, SHAs and source-field locators are runtime parameters, never tracked
literals.

Selection is immutable SHA-256 ordering followed by a global one-row-per-
component and one-row-per-exact-source rule. It takes exactly:

1. 100 `direct_included_fuel` rows;
2. 100 `device_accessory_compatible_mention` rows;
3. 100 `singleton_new_family_ambiguous` rows.

The teacher request strips stratum and all row/component/family tokens. It sees
only an audit ID, an opaque record ID, the source fields, deterministic evidence
candidates and cryptographic request binding.

## Frozen human-audit gate

The frozen packet is copied for independent review; only the four `review`
booleans may change. `evidence_supported=true` means either that selected
evidence supports the structured target or that an unsupported/ambiguous
abstention is correct. The preferred result is zero unsupported or incorrect
evidence, while the hard gate permits at most three.

All conditions must hold:

- exact schema and mechanical grounding `300/300`;
- joint `sold_object + relation` correctness at least `282/300`;
- joint correctness at least `95/100` in the critical
  device/accessory/compatibility/mention stratum;
- evidence supported at least `297/300`;
- `supervise=true` coverage at least `225/300` overall and `80/100` in the
  direct/included-fuel stratum;
- contradictions at most `15/300`;
- exact record/source nonduplication, unique selected components and zero
  row/component/family overlap with the 670/672 exclusions.

Even an accepted audit emits `READY_FOR_SEPARATE_STUDENT_GPU_GO` with
`student_gpu_authorized=false`. A fresh independent gate must later freeze the
student parent, data split, objective weight and paired control before any GPU
work. No target is currently approved for training.

## Exact remote commands

The following commands are templates for a future CPU-only remote compute runtime. All
paths are runtime parameters and must resolve below the same remote root.

```bash
python3 experiments/689_qwen35_4b_grounded_transaction_graph_kd/build_target_audit.py prepare \
  --remote-root /work/exp689 \
  --source-rows /work/exp689/inputs/source_rows.jsonl \
  --source-contract /work/exp689/inputs/source_contract.json \
  --exclusion-670 /work/exp689/inputs/exclusion_670.csv \
  --exclusion-672 /work/exp689/inputs/exclusion_672.json \
  --exclusion-670-sha256 "$EXP670_SHA256" \
  --exclusion-672-sha256 "$EXP672_SHA256" \
  --exclusion-670-adapter csv_component_column \
  --exclusion-672-adapter json_component_list \
  --exclusion-670-component-locator "$EXP670_COMPONENT_COLUMN" \
  --exclusion-672-component-locator "$EXP672_COMPONENT_LOCATOR" \
  --exclusion-672-component-value-field "$EXP672_COMPONENT_VALUE_FIELD" \
  --exclusion-670-token-mode sha256_utf8_v1 \
  --exclusion-672-token-mode sha256_utf8_v1 \
  --output-dir /work/exp689/outputs/prepared
```

The teacher runs against only `prepared/teacher_request.jsonl` and must produce
contract-bound closed selections. Materialization is then:

```bash
python3 experiments/689_qwen35_4b_grounded_transaction_graph_kd/build_target_audit.py materialize \
  --remote-root /work/exp689 \
  --selection-manifest /work/exp689/outputs/prepared/selection_manifest.json \
  --teacher-request /work/exp689/outputs/prepared/teacher_request.jsonl \
  --teacher-selections /work/exp689/inputs/teacher_selections.jsonl \
  --teacher-selection-contract /work/exp689/inputs/teacher_selection_contract.json \
  --output-dir /work/exp689/outputs/frozen_audit
```

After a separate reviewer fills only the review booleans in a distinct copy:

```bash
python3 experiments/689_qwen35_4b_grounded_transaction_graph_kd/validate_target_audit.py \
  --remote-root /work/exp689 \
  --source-rows /work/exp689/inputs/source_rows.jsonl \
  --source-contract /work/exp689/inputs/source_contract.json \
  --selection-manifest /work/exp689/outputs/prepared/selection_manifest.json \
  --teacher-request /work/exp689/outputs/prepared/teacher_request.jsonl \
  --teacher-selections /work/exp689/inputs/teacher_selections.jsonl \
  --teacher-selection-contract /work/exp689/inputs/teacher_selection_contract.json \
  --frozen-packet /work/exp689/outputs/frozen_audit/target_audit.jsonl \
  --completed-review /work/exp689/inputs/completed_review.jsonl \
  --packet-contract /work/exp689/outputs/frozen_audit/target_audit_contract.json \
  --exclusion-670 /work/exp689/inputs/exclusion_670.json \
  --exclusion-672 /work/exp689/inputs/exclusion_672.json \
  --output /work/exp689/outputs/target_audit_acceptance.json
```

Every output path is immutable: an existing output directory or result is a
hard failure. No local fallback, implicit data discovery, preset generation,
artifact upload or GPU launch exists in these entrypoints.
