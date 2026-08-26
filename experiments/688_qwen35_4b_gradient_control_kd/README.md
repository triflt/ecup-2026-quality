# Experiment 688 — hard-primary gradient-controlled rank KD

Status: `TERMINAL_NO_GO_SELECTOR`. Parent: experiment 686.

## Terminal selector result

Experiment 687 returned the frozen decision
`REJECT_PCGRAD_LOW_RETAINED_RANK_SIGNAL`. Experiment 688 permits a candidate
only for `OPEN_ASYMMETRIC_PCGRAD_SCREEN` or `ROUTE_MAGNITUDE_CONTROL`, so no
mode is selected and no bundle, preset, upload or GPU job is created. This is a
terminal negative result, not an infrastructure block. Manually choosing
PCGrad, norm-cap, a different lambda or a decision threshold would violate the
preregistered selector and is forbidden.

No generated preset, upload or job is authorized or present. Infrastructure-only
builders remain unused because the terminal, independently accepted
experiment-687 artifact selected no candidate.

## Hypothesis and one changed factor

Experiment 686 established the frozen additive objective
`hard_BCE + 0.5 * pair_rank_BCE`. Experiment 688 changes only how the two
effective-batch gradients are composed. It does not change the teacher, targets,
pairs, model, LoRA, initialization, order, seed, optimizer, learning rate,
scheduler, number of steps, preprocessing or decision threshold.

For every exact effective batch of eight ordered pairs (16 rows), the runtime
uses one shared forward graph per pair and the same two-backward accumulator in
all arms. It accumulates the mean hard gradient `h` and the mean *unweighted*
rank gradient `r` separately. The paired control discards `r` only after both
gradients have been computed, so its pre-clip applied gradient is exactly `h`
through the same autograd path.

There are exactly two preregisterable candidate modes:

1. `asymmetric_hard_primary_pcgrad` is selected only by terminal experiment-687
   decision `OPEN_ASYMMETRIC_PCGRAD_SCREEN`. If `h·r < 0`, only the rank gradient
   is projected: `r' = r - (h·r / ||h||²) h`; otherwise `r' = r`. The applied
   pre-clip update is `h + 0.5 r'`. The hard gradient is never projected before
   the common global-norm clip; no claim is made that the clip preserves its
   absolute magnitude.
2. `hard_anchored_norm_cap` is selected only by terminal experiment-687 decision
   `ROUTE_MAGNITUDE_CONTROL`. Let `u = 0.5 r`; scale `u` by
   `min(1, 0.5||h|| / ||u||)`, then apply `h + u`. Thus the rank contribution
   cannot exceed half the hard-gradient norm.

All other terminal experiment-687 decisions fail closed and select no experiment-688
candidate. There is no lambda, weight, cap or threshold grid: rank weight and cap
ratio are both frozen at `0.5`.

## Frozen parent contract

- model: `Qwen/Qwen3.5-4B`, immutable revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- rsLoRA: rank 16, alpha 32, dropout 0.05, bias none, q/k/v/o projections;
- exact experiment-686 source runtime, flammable rows, teacher pair runtime,
  pair targets and ordered occurrence bindings;
- 5,440 pairs, one epoch, shuffle seed 42, eight pairs per optimizer update,
  680 updates, micro-batch two rows and effective batch 16 rows;
- AdamW, learning rate `2e-4`, weight decay `0.01`, 5% linear warmup followed by
  the same cosine decay, pre-step global gradient clipping at `1.0`;
- legacy-eager backend, identical image/prompt preprocessing and threshold `0`;
- zero training reads of outer-validation labels, sealed rows or Public data.

Any drift in those fields is a stop condition, not a new exp688 run.

## Diagnostic and artifact contract

Every optimizer step emits one `gradient_diagnostics.jsonl` record with exact
schema and finite values for the hard/rank dot product, cosine, both norms,
weighted rank norm, conflict flag, hypothetical asymmetric-projection retention,
norm-cap limit/scale/application, selected rank retention, theoretical combined
pre-clip norm and the actual value returned by `clip_grad_norm_`. The verifier
requires the actual and theoretical pre-clip norms to agree. Hard or rank norm
zero, NaN/Inf, missing parameter gradients, a batch other than eight pairs, or a
diagnostic formula mismatch fails closed.

The output inventory is exactly the self-hashed `output_contract.json`, the
diagnostic JSONL, label-free validation predictions and the standard PEFT adapter
files. `verify_training_artifact.py` rebinds the terminal exp687 report and
acceptance, parent runtimes, transport, model/LoRA contract, pair order, code and
vendor SHAs, diagnostics, predictions, save/reload smoke parity and data-boundary
counters before emitting acceptance.

## Gates

1. Wait for terminal exp687 plus its independent `ACCEPT_GRADIENT_CONFLICT_PROBE`.
2. Select exactly the mapped candidate above; no human override or fallback mode.
3. Independently review this source and freeze a code bundle/revision. Only then
   may a paired control/candidate technical smoke be proposed. The sole current
   selector lineage is fold3. Smoke runs control then the selected candidate in
   fresh sequential processes on one H100, using the same first eight positions
   of the full seed-42 shuffled pair order, one optimizer step per arm, two
   label-free validation rows and two separate output directories.
4. A live scientific fold requires a fresh explicit GO after smoke acceptance.
   Candidate and control must share exact source/pair/model/code/runtime bindings.
5. Evaluation, refit, packaging and Public remain closed until a separately
   reviewed remote-first evaluator and promotion gate accept the required paired
   artifacts. Technical success is not quality evidence.

Relevant precedent is PCGrad (Yu et al., NeurIPS 2020): conflicting auxiliary
task gradients can be projected before a joint update. This experiment uses the
more conservative asymmetric variant because hard labels are the proven primary
task; the magnitude-control alternative tests the distinct diagnosis that rank
gradient dominance, rather than negative alignment, destabilizes the boundary.

## Entrypoints

`build_code_bundle.py` and `verify_code_bundle.py` create and accept the exact
runtime whitelist. `build_remote_compute_preset.py` validates terminal exp687 and the
parent/probe/exp688 code acceptances before writing only a secret-free clean
preset; region, bucket and approved prefix come from ignored local configuration.
Credentials never enter this tracked builder. `verify_paired_smoke.py` binds the
two remote arms and rejects code/data/model/init/order/runtime or raw-gradient
drift. `train_gradient_control.py --help` exposes the one-step
`--technical-smoke` path. No preset instance is built by this packet. The local
submit path must inject credentials only in memory through stdin and must never
persist, print or hash the secret-bearing payload.

The current fold3 smoke has one explicit parent-transport adapter. It accepts
only the exact terminal experiment-687 lineage for the frozen experiment-686
`9899e20` parent bundle and fold3 pair contract, then invokes that parent's
legacy `stage_training_input.py` interface. It does not pass the later R0-code
acceptance option, which that immutable parent does not implement. No generic
fallback to another parent revision, pair contract or source fold is allowed.
