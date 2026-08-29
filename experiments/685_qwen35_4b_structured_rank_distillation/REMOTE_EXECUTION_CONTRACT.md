# remote compute / S3 execution contract

Status: frozen before the first 685 live job.

## Purpose

Corporate datasets, predictions, teacher scores, pair packets, adapters and
model outputs and terminal reports stay in approved remote compute/S3 storage. The
local workstation is a control plane for reviewed code, immutable remote
references, job receipts, URIs/SHA and concise human-readable summaries. It is
not an artifact-processing environment.

## Data flow

0. The accepted legacy 662 fold0/fold3 remote compute outputs are copied once by a
   label-free bridge job from exact `JOB/OUTPUT` inputs to a new immutable S3
   prefix. The bridge accepts only the canonical inner archive with frozen
   archive and teacher-score SHA; loose diagnostic files cannot substitute for
   it. All later 685 stages consume the S3 copy.
1. Each input is addressed by a unique S3 key plus SHA-256. Existing keys are
   immutable and are never overwritten.
2. An remote compute preparation job reads accepted source artifacts from approved
   S3, builds the fold-specific 685 pair runtime, verifies it in the same job
   and writes the runtime plus acceptance report to a new S3 prefix.
3. Training jobs read only an accepted pair-runtime key and its frozen SHA.
   Control and candidate use the same runtime, model revision, seed, batches,
   steps and validation rows. The sole changed factor is the rank term.
   Images retain the proven 641 row-level `image_url` semantics and use only an
   ephemeral per-job cache. Adding a new persistent image-cache artifact would
   be a separate experiment and is forbidden in 685A.
4. Every training job computes label-free structural/runtime checks before exit
   and writes predictions, adapter and a self-hashed result contract to S3.
5. Performance checks run as separate minimal remote compute eval jobs. They alone
   receive the frozen label packet, read predictions by S3 key and emit a
   compact self-hashed metrics JSON. Training code cannot read validation
   labels.
   For the outer0 promotion decision, a prebuilt CPU fast evaluator constructs
   a full 943-row immutable label packet only inside remote compute from the frozen
   semantic-v3 registry (SHA-256
   `16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae`).
   It binds the packet to fold0 validation by exact ordered
   `global_index/id/fold/category` and compares the paired control and rank
   candidate directly. The sampled fold3 training runtime is explicitly
   forbidden as a label donor because it does not cover every validation row.
   Because BAD is frozen byte-identical,
   its exact routed Macro delta is one half of the flammable F1 delta. This fast
   report is authoritative only for `OPEN_SCREEN_FOLD3`; later screen/full gates
   still require the complete frozen replay evaluator.
6. No job output artifact, including compact metrics JSON, is downloaded to the
   workstation. Terminal values may be inspected through logs/API and recorded
   as a summary plus remote URI/SHA. A full ZIP is downloaded only for a
   candidate that passed scientific, package and runtime gates and is ready for
   submission, after fresh explicit user approval.

## Eval placement

Run inside every training job without validation labels:

- finite/output-schema and exact row-binding checks;
- peak GPU memory, throughput and ETA telemetry;
- adapter save/reload parity and provenance hashes.

Run as a separate CPU eval job for each accepted prediction set:

- flammable AP, F1, FP/FN and calibration diagnostics;
- routed Macro F1 with byte-identical BAD predictions;
- corrections/regressions and semantic-singleton slices;
- paired control-versus-candidate bootstrap;
- five-fold aggregate and stability gates;
- error-component attribution or large slice joins;
- cross-check using an evaluator not packaged with the trainer;

Use a separate minimal-GPU job only for production package/runtime rehearsal
or an evaluator that genuinely performs model inference.

The evaluator never selects thresholds, weights or hyperparameters using an
outer validation fold. Frozen threshold zero remains unchanged in 685A.

## Latency controls

- Preparation, training and evaluation use content-addressed inputs, so an
  accepted pair packet or prediction set is reused without regeneration.
- Label-free checks are fused into training. Performance eval jobs default to
  CPU; one GPU is allowed only when evaluator
  inference genuinely requires it.
- Control and candidate are submitted in the same wave. With the current
  project cap of 8 GPUs, at most 6 are working and 2 remain reserved.
- Fold3 control/candidate may use two otherwise idle GPUs as sealed speculative
  compute only when their immutable presets and output keys were frozen before
  any outer0 quality result. Until outer0 promotion, monitoring is state-only;
  fold3 predictions, acceptance reports and metrics are unread. Outer0 reject
  makes those outputs permanently unused. This exception never applies to
  folds1/2/4.
- Polling is event-driven with a 25-minute minimum interval. ETA comes from
  observed rows/second and p90 batch time, not calendar estimates.

Expected overhead is one small S3 read/write per stage plus remote compute queue time.
For multi-minute training this is minor; it is smaller than the time lost to
local downloads, DLP failures and repeated verification.

## S3 authorization gate

Every preset uses exactly one of two approved modes:

1. a bucket with native remote compute integration in the job tenant; or
2. cross-tenant S3 credentials referenced from Vault with an exact
   `vault.auth_role`, access-key reference and secret-key reference.

The three Vault settings are all-or-nothing. Literal credentials are forbidden
in source, git, shell history, presets and chat. A missing native integration or
an incomplete Vault configuration is a pre-submit transport failure and must
not fall back to local downloads, managed artifact outputs or guessed storage
names. The current approved external bucket failed the native-integration
precondition before job creation; no 685 scientific job may start until one of
the two modes above is proven by a dry-run and a minimal remote-only bridge.

## Security and failure policy

Corporate DLP/EDR/quarantine/provenance controls are never bypassed. On
`Operation not permitted`, `Permission denied` or any sign of a protection
block, stop accessing and transforming that object, preserve it and its
metadata, and report the event. Do not remove provenance/quarantine metadata,
change permissions or flags, copy, rename, repack, or retry with another tool.

Before any local download of a corporate archive containing data,
predictions/model outputs, or larger than 50 MB, obtain fresh explicit user
approval. Corporate artifacts may be written only to the approved project
storage. Transport and scientific failures are classified separately; an
exact retry is allowed only for a proven transport-only failure with unchanged
scientific inputs and contracts.
