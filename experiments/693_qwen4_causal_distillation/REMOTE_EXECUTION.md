# Three-job remote execution contract

`build_code_bundle.py build` creates one deterministic, Git-revision-bound
archive from this exact whitelist:

- experiments 693, 694 and 695;
- `645_qwen_scale_2x3_gate/grid_contract.py`, `train_lora.py` and its frozen spec;
- the frozen experiment-641 spec.

`build_preset.py` creates exactly `qwen4-causal.yaml`, `qwen4-hardneg.yaml` and
`qwen4-rank.yaml`. Project, region, image, H100 flavor, buckets, every S3 prefix,
private model MRID and immutable archive identities are required arguments. No
private default is tracked. The MRID must end in Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Project is validated by the builder but is not a preset-schema field; a future
authorized submitter must pass that same value through the CLI `-p` argument.

Frozen archive layouts are:

- runtime bundle: `runtime/fold0` through `runtime/fold4`;
- baseline bundle: `fold0/predictions.jsonl` through fold4 at archive root;
- PEFT vendor bundle: `peft/__init__.py` at archive root;
- selected teacher terminal output: experiment 691 uses `fold0` through `fold4`
  at the mounted root; experiment 696 uses `fivefold/fold0` through
  `fivefold/fold4`;
- exp692 outputs: the exact routed acceptance and winner gate named by the
  builder arguments.

Each 1×H100 job first runs one changed-factor technical smoke, then trains only
the candidate sequentially on folds 0..4 and evaluates it against the accepted
frozen production route. Each fold trains outside the S3 output root and becomes
visible at `candidate/foldK` by an atomic rename only after success. The S3
output has deliberately no `upload_policies`, so remote compute uploads completed fold
directories and the minimal failure marker on any terminal state. No builder
uploads, dry-runs or submits.

Final `evaluation.json` is method-bound and self-hashed. It binds the exact
teacher acceptance and winner file/self hashes, runtime and baseline bundle
hashes, and an ordered list of five candidate output-contract/prediction
bindings. Full candidate provenance is checked before cross-fold labels open.
Candidate training runtime and peak memory remain observations only: the job time
limit is never passed as a deployment-inference limit. Consequently a
science-passing candidate stays `STAGE_PENDING_RESOURCE_EVIDENCE` until a later
package inference measurement supplies the real submission-limit evidence.
