# Experiment 688 — hard-primary gradient-controlled rank KD

Status: `PREREGISTERED_AWAITING_TERMINAL_EXP687`. Parent: experiment 686.

No preset, upload or job is authorized or present. A terminal, independently
accepted experiment-687 artifact must select exactly one candidate before even a
technical smoke can be assembled.

## Hypothesis and one changed factor

Experiment 686 established the frozen additive objective
`hard_BCE + 0.5 * pair_rank_BCE`. Experiment 688 changes only how the two
effective-batch gradients are composed. It does not change the teacher, targets,
pairs, model, LoRA, initialization, order, seed, optimizer, learning rate,
scheduler, number of steps, preprocessing or decision threshold.

For every exact effective batch of eight ordered pairs (16 rows), the runtime
uses one shared forward graph per pair and the same two `autograd.grad` calls in
all arms. It accumulates the mean hard gradient `h` and the mean *unweighted*
rank gradient `r` separately. The paired control discards `r` only after both
gradients have been computed, so its applied update is exactly `h` through the
same autograd path.

There are exactly two preregisterable candidate modes:

1. `asymmetric_hard_primary_pcgrad` is selected only by terminal experiment-687
   decision `OPEN_ASYMMETRIC_PCGRAD_SCREEN`. If `h·r < 0`, only the rank gradient
   is projected: `r' = r - (h·r / ||h||²) h`; otherwise `r' = r`. The applied
   pre-clip update is `h + 0.5 r'`. The hard gradient is never projected.
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
norm-cap limit/scale/application, selected rank retention and combined pre-clip
norm. Hard or rank norm zero, NaN/Inf, missing parameter gradients, a batch other
than eight pairs, or a diagnostic formula mismatch fails closed.

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
   may a paired control/candidate technical smoke be proposed. Smoke uses exactly
   the first shuffled eight pairs, one optimizer step, two label-free validation
   rows, adapter save/reload, finite diagnostics and one H100 per arm.
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

`train_gradient_control.py --help` exposes the train and `--technical-smoke`
path. `verify_training_artifact.py --help` exposes independent artifact replay.
No remote compute preset is built until terminal exp687 selects one candidate.
