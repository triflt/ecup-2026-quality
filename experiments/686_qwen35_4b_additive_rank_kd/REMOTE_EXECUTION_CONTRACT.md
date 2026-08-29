# Remote execution contract — experiment 686

Status: frozen before the first 686 GPU job.

## Inputs

Experiment 686 reuses only accepted, immutable experiment-685 inputs:

- fold-specific pair runtime and exact-whitelist transport acceptance;
- frozen 641 source runtime embedded in that accepted delivery;
- vendored PEFT 0.20.0 bridge;
- Qwen3.5-4B model-registry revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

Every consumer verifies the recorded archive, runtime, acceptance, source and
model-tree hashes before model loading. Loose diagnostic files cannot replace an
accepted archive. Experiment 686 does not rebuild teacher targets and does not
read cancelled experiment-685 fold-3 outputs.

## Physical bundle separation

Two immutable code bundles are required:

1. `scope=training` contains the 686 trainer/verifiers and the frozen 641 train
   implementation. It must not contain `validation/semantic_family_v3/folds.csv`
   or frozen replay label artifacts.
2. `scope=evaluation` contains the training sources plus the semantic-v3 registry
   and replay evaluators. It is used only after both training outputs are terminal
   and accepted.

The code verifier records and self-hashes the scope. Training refuses any code
acceptance whose scope is not exactly `training`.
The preset generator also consumes that local self-hashed acceptance before job
creation, pins its archive SHA/revision, and passes `--expected-scope training`
to the remote verifier. Evaluation analogously requires `scope=evaluation`.

## Paired training

The blind screen is fold 3. Exactly two 1×H100 jobs are allowed:

- control: `hard_BCE`;
- candidate: `hard_BCE + 0.5 * pair_rank_BCE`.

The pair term is BCE on the positive-minus-negative verdict logit difference,
with the frozen teacher-derived pair target. Both jobs share the same base model,
LoRA configuration, train rows, pair multiset and order, initialization digest,
seed 42, LR `2e-4`, micro-batch two rows, accumulation eight pairs, effective
batch 16 rows, one epoch, 680 optimizer steps and threshold zero. The rank-loss
term is the only allowed scientific difference.

Training output is written to a new S3 prefix only after job success. It contains
adapter, predictions, self-hashed contract and independent acceptance. Training
must report validation-label reads 0, sealed rows 0 and Public false.

## Evaluation and promotion

The compact remote CPU evaluator binds the accepted fold-3 predictions to the
exact 943-row semantic-v3 packet only as a technical smoke. It verifies full
packet coverage but cannot promote. The scientific blind decision is made by the
separate frozen full-production replay, including byte-identical BAD checks.

The only first-stage decisions are:

- `OPEN_CONFIRMATION_FOLDS124`; or
- `REJECT_AT_OUTER3`.

Fold-3 promotion requires all frozen gates in `README.md`. Until promotion,
folds 1/2/4, refit, package and Public are closed. Fold 0 is development-only and
must not be presented as independent confirmation.

Folds 1/2/4 require the independently issued fold-3
`OPEN_CONFIRMATION_FOLDS124` promotion gate both when the preset is generated and
inside each remote training job. The strict gate is bound to the terminal remote compute
job identity/state, immutable output source, job-metadata SHA, original
evaluation bytes/self-hash, code/replay/registry hashes, exact input artifact
hashes and an all-true gate map. Training mounts and verifies both
`promotion_gate.json` and the original `evaluation.json`; their raw hashes enter
artifact provenance.
The confirmation evaluator requires the same receipt. Fold 0 and the full replay
remain closed until the analogous self-hashed `OPEN_FULL_REPLAY` confirmation
receipt exists. Blind fold 3 is forbidden from consuming any promotion receipt.

## Security and operational rules

- Use project name only from `.local/compute-project.txt`.
- Use only the approved `biglm-alignment-pipeline/d.strizhakov/ecup` prefix.
- Keep job artifacts remote; no local artifact ZIP download before a qualified
  submission build with explicit user approval.
- Maintain an owned-job registry; at most eight user-owned GPU jobs may run.
- Before submission: preset audit, dry-run, immutable bundle SHA and new-output
  prefix checks must pass.
- On `Operation not permitted`, `Permission denied`, DLP/EDR/quarantine or
  provenance blocking, stop access to that object immediately. Do not retry with
  another tool, remove metadata, chmod, copy, rename, repack or bypass controls.
