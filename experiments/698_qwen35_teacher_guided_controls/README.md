# 698: Qwen3.5-4B teacher-guided matched controls

This experiment consumes the exact five frozen experiment-697 outer-fold
runtimes and occurrence-aligned frozen 27B teacher targets. It deliberately does
not pass those artifacts through the older 691/696 consumer: that consumer uses
a different 11,118-row semantic-family runtime and cannot truthfully bind the
12,971-row `competition_train_v1` experiment-697 protocol.

Three Qwen3.5-4B arms share the same model, seed, prompt, runtime occurrence
multiset, order, optimizer, LoRA structure and effective batch:

- `gold_control`: hard-gold BCE only and does not read teacher targets;
- `hardneg_candidate`: hard-gold BCE with bounded weights only for flammable
  rows on which the frozen teacher disagrees with gold;
- `rank_candidate`: hard-gold BCE plus within-batch ranking only between
  flammable rows in the same hard-label stratum.

No soft teacher verdict replaces gold. No opposite-label rank pair is allowed.
The 27B model and its targets are training-only; every output adapter uses the
competition-provided Qwen3.5-4B and remains compatible with replacing the
Qwen3.5 adapter pass in solution 140.

All three student arms use the exact solution-140 Qwen3.5 image contract:
the first RGB image is resized in place with Pillow `thumbnail((448, 448),
LANCZOS)`. This intentionally differs from the offline 27B teacher's larger
area-bounded view: knowledge may cross resolution during training, but student
OOF and deployment preprocessing must be identical. The preprocessing version
is required in every fold output contract and downstream package contract.

`queue_after_teacher.sh` contains two target-generation lanes and one student
lane. It is safe to start while the teacher is running: each lane waits on
non-empty immutable contracts, skips completed outputs, and refuses to overwrite
partial outputs. Five student folds run concurrently on five GPUs, but each
individual train/inference process uses one H100.

The evaluator verifies all fold contracts, self-hashes, prediction checksums and
exact ID coverage. The v2 fold contract also binds the base model, seed,
occurrence order, sampler, optimizer, scheduler, LoRA configuration, adapter
checksums and preprocessing. Evaluation fails unless those common factors and
each fold runtime are identical across all three arms, both candidate arms bind
the exact same teacher targets, and the gold control binds none. Before nested
threshold selection it converts each adapter's
raw logits to label-blind percentile ranks separately inside every `(outer
fold, category)` stratum. This prevents fold-specific logit offsets or scales
from creating a false cross-fold gain and matches solution 140's rank-based
runtime fusion. Both raw and calibrated OOF scores are retained. A distillation method
passes the initial science gate only with Macro delta at least +0.001 versus the
matched gold control, at least 4/5 fold wins and no category regression below
-0.005. Component replacement/fusion and a full-data adapter are later stages,
not inferred from standalone student scores.

For an optional standalone Qwen3.5 deployment, each category also records a
rank threshold equal to the median of the five outer-train donor thresholds.
It is never used to score the corresponding nested OOF fold and is not allowed
to influence promotion; it is only a conservative production setting for a
full-data adapter after all gates have passed.

`connected_guard.py` is a second fail-closed promotion stage. It replays the
frozen nested predictions on the 11,207 `connected_family_guard_v2` safe rows,
bootstraps category-specific connected components, and requires both the primary
OOF gate and the connected-safe gate. This is specifically intended to catch
the kind of local improvement that failed to transfer in experiment 230.
