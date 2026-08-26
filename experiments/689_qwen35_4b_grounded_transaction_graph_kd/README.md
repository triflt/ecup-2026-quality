# Experiment 689 — image-grounded transaction-graph KD

Status: `PREREGISTERED_REMOTE_TARGET_AUDIT_ONLY`. Student GPU: `NO_GO`.
Jobs, uploads, presets and bundles: `0`.

## Hypothesis and frozen target

The preregistered hypothesis is that a grounded transaction graph can transfer
the distinction between what is sold, what substance is present, and how the
item relates to that substance without copying free-form teacher reasoning or
outcome scores. This packet freezes only the 300-row target audit. It does not
select or authorize a student-training parent.

The closed target enums are:

- `sold_object`: `device`, `fuel_consumable`, `accessory`, `kit`, `other`,
  `unknown`;
- `substance`: `gas`, `flammable_liquid`, `solid_fuel`, `ignition_aid`, `none`,
  `unknown`;
- `relation`: `primary_sold_object`, `included`, `compatible_external`,
  `mentioned_only`, `negated`, `absent`, `unknown`.

Teacher output contains no free text, confidence, verdict, label, logit,
probability, rank, family ID or chain-of-thought. Each of `sold_object`,
`substance`, and `relation` binds its own sorted list of existing deterministic
evidence candidate IDs. A supported row requires a nonempty list for all three
targets and `supervise=true`. `unsupported` or `ambiguous` requires all three
lists empty and `supervise=false`; no code or evidence may be invented.

## Image-aware source contract

Every source row includes its first-image reference, encoded-content SHA,
decoded-RGB SHA and decoded dimensions. The teacher request includes that image
binding plus deterministic candidates for the full decoded image and its four
2×2 quadrants, alongside exact text/OCR spans. A text/OCR-only scientific packet
is forbidden.

The tracked prepare stage does not fetch images. It accepts only source rows
whose immutable source contract says all 300 first images were verified and
that image fetch, decode and hash failures are all zero. Any future remote
source runner must stop on any such failure. The source contract also binds
runtime, data, registry, builder revision, eligibility universe, stratum
derivation, first-image membership, image transform, and candidate-generator
SHAs.

The teacher-selection contract binds request/output SHAs, model ID and revision,
prompt, decoding, inference bundle, and job metadata. Model and runtime
references remain caller-supplied; tracked files contain no account, bucket,
endpoint, credential, personal path or internal artifact literal. This commit
does not provide teacher smoke/full job builders.

## Deterministic sample and exact claim

Prepare is a separate CPU-only remote stage. It selects 100 rows from each
stratum by frozen SHA ordering, with one selected row per component and exact
source:

1. S2 — `direct_included_fuel`;
2. S1 — `device_accessory_compatible_mention`;
3. S3 — `singleton_new_family_ambiguous`.

Runtime exclusion sources must contain 300 and 40 unique component identities,
respectively, with zero intersection and union size 340. The selected packet
must have zero component overlap with that union. This packet deliberately
claims only component disjointness from experiments 670/672; it does not claim
family disjointness without corresponding evidence.

## Dual-blind review and adjudication

The frozen target packet contains no review fields. Reviewer A and reviewer B
produce separate immutable 300-row overlays. Each self-hashed reviewer contract
binds the packet, review rubric, overlay, opaque reviewer ID, start/completion
timestamps, row count, and blindness to the other reviewer, outcome labels, and
prior audits. Reviewer A and reviewer B must be different actors.

Each overlay records the six review flags plus exactly one evidence slice:
`text_sufficient`, `image_helpful`, or `image_required`. If any row differs
between reviewers, a distinct adjudicator must provide an overlay containing
exactly all disagreement rows and no others. Adjudication is forbidden when
there are no disagreements.

All frozen gates must pass:

- schema and grounding `300/300`;
- strict target tuple (`sold_object + substance + relation`) at least `282/300`;
- object+relation at least S1 `95/100`, S2 `90/100`, and S3 `90/100`;
- supported evidence at least `297/300`, coverage at least `225/300` overall
  and `80/100` in direct/included fuel, contradictions at most `15/300`;
- exact whole-review agreement at least `0.90`, relation Cohen kappa at least
  `0.80`, and relation Gwet AC1 at least `0.80`; both imbalance-sensitive and
  skew-robust agreement gates must pass;
- at least 30 final `image_required` rows, at least 27 strict passes in that
  slice, and zero unsupported visual claims there;
- exact record/source nonduplication and the component-disjointness claim above.

Strict row pass means all three target fields correct, evidence supported, no
contradiction, and no unsupported visual claim. Even acceptance emits only
`READY_FOR_SEPARATE_STUDENT_GPU_GO` with `student_gpu_authorized=false`; a fresh
independent gate is still required.

## Remote command surfaces

All paths below are examples supplied by the runtime and must resolve under the
same `--remote-root`. Prepare consumes already verified image-aware source rows:

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

Teacher inference is external to this commit. After its exact output and
contract exist, materialization is:

```bash
python3 experiments/689_qwen35_4b_grounded_transaction_graph_kd/build_target_audit.py materialize \
  --remote-root /work/exp689 \
  --selection-manifest /work/exp689/outputs/prepared/selection_manifest.json \
  --teacher-request /work/exp689/outputs/prepared/teacher_request.jsonl \
  --teacher-selections /work/exp689/inputs/teacher_selections.jsonl \
  --teacher-selection-contract /work/exp689/inputs/teacher_selection_contract.json \
  --output-dir /work/exp689/outputs/frozen_audit
```

After two blind overlays, and optionally exact disagreements-only adjudication:

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
  --review-a /work/exp689/inputs/review_a.jsonl \
  --review-a-contract /work/exp689/inputs/review_a_contract.json \
  --review-b /work/exp689/inputs/review_b.jsonl \
  --review-b-contract /work/exp689/inputs/review_b_contract.json \
  --packet-contract /work/exp689/outputs/frozen_audit/target_audit_contract.json \
  --exclusion-670 /work/exp689/inputs/exclusion_670.csv \
  --exclusion-672 /work/exp689/inputs/exclusion_672.json \
  --output /work/exp689/outputs/target_audit_acceptance.json
```

When disagreements exist, both `--adjudication` and
`--adjudication-contract` are additionally required. Existing outputs are never
overwritten. There is no local fallback, implicit data discovery, preset
generation, artifact upload or job launch in these entrypoints.
