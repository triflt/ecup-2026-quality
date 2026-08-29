# Large open teacher value audit

Date: 2026-08-22
Status: research recommendation only; no model was launched and no shared experiment log was changed.

## Decision

Use `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` only for a small, blinded
teacher-versus-2B pilot. Its highest-probability value is **evidence-grounded
silver rationale generation**, followed by visual/OCR evidence extraction and
label-audit triage. A direct F1 gain is the least certain use and must not be the
reason to fund a full-dataset pass.

This model is a realistic teacher because it is an official Apache-2.0 Qwen
checkpoint, has 236B total parameters (below the 400B cap), 22B active MoE
parameters, and an official FP8/vLLM recipe for tensor parallelism across eight
H100 GPUs. It remains strictly an offline training/audit dependency; the final
submission continues to run the allowed small model.

Do **not** use a thinking checkpoint or save free-form chain of thought. Request
short, typed evidence objects and a concise conclusion. The desired artifact is
verifiable evidence, not hidden reasoning.

## Why this is permitted, and the boundary

The local rules say that external datasets are forbidden, while synthetic
examples or automatic annotations from an appropriately licensed open model up
to 400B are allowed if the generation method and generated data are saved and
the organizer can reproduce generation on at least several thousand examples.
The rules also require final comments of 50--300 characters and state that
concrete card-specific explanations are manually assessed; generic comments are
insufficient. See [`docs/hackathon/task-and-rules.md`](../../docs/hackathon/task-and-rules.md).

Therefore the teacher may receive only competition-provided training text and
images. No web search, retrieval corpus, product catalogue, private label guide,
or third-party OCR corpus may enter its prompt or its generated dataset.
Pretraining already embodied in the permitted open checkpoint is not an
additional dataset supplied by this pipeline.

For reproducibility, preserve the exact model revision, container digest, vLLM
version, prompt/schema, decoding parameters, input row/image hashes, raw model
output, parsed output, validator decisions, and deterministic sample manifest.
Never call the teacher on hidden test data and never package its weights in the
submission.

## What our evidence says

The current category-routed Qwen3.5 classifier is already strong: locked OOF
Macro F1 is 0.931645, with BAD 0.954660 and flammable 0.908629. Its 538 residual
errors comprise 502 BAD errors and 36 flammable errors. Of the BAD errors, 122
(24.3%) are in mixed-label connected components; all 36 flammable errors are in
connected-safe families. These cohorts include non-intuitive annotation
contrasts such as the sold item versus an included component, lighter/matches,
liquid fuel, solid fuel, and empty equipment. Sources:
[`residual_error_report.json`](../../experiments/400_qwen35_category_routed_adapters/analysis/residual_error_report.json)
and [`agent_hard_errors/REPORT.md`](../../experiments/290_minicpm_v46_visual_screen/analysis/agent_hard_errors/REPORT.md).

The current explanation renderer is not evidence grounded. It sees only
category and predicted label and chooses one of four generic strings; it cannot
cite the name, description, image, OCR span, or sold component. This has been
recorded in
[`explanation_readiness_audit.json`](../../experiments/400_qwen35_category_routed_adapters/analysis/explanation_readiness_audit.json).
Thus explanation quality is an independent, real weakness even if classification
F1 is unchanged.

Direct small-model visual routes have not provided F1 evidence:

- MiniCPM-V-4.6 first-image LoRA: folds 0/3 mean Macro delta -0.041895 and
  flammable delta -0.079607.
- InternVL3.5-2B zero-shot attributes: mean Macro delta -0.006572; 336/909
  outputs were invalid, and regressions exceeded corrections 9 to 4.
- Qwen3-VL-2B R-Drop: mean Macro delta -0.020291.
- Qwen3-VL-Embedding-2B: mean Macro delta -0.016656 and flammable delta
  -0.033311.

These results do not prove that every small VLM is incapable of grounding.
They do show that another direct small-model verdict branch is not justified and
that structured-output validity must be a first-class gate.

## Value by use case

### 1. Classification F1: low-confidence upside

A large generic teacher understands product imagery better, but the target is
the organizers' annotation function, not ordinary semantic truth. Connected
families with identical accessible inputs and different labels contain no
row-specific signal for any teacher to recover. For the remaining cases, a
teacher may still prefer commonsense hazard semantics over the observed
non-intuitive label ontology. Direct teacher labels or pseudo-label replacement
are therefore unsafe.

The only defensible F1 mechanism is **rationale-guided student training**:
generate evidence/concept targets for training rows, then add one auxiliary
evidence/concept objective to the existing small student while keeping its gold
classification target and recipe unchanged. Distillation research supports the
general mechanism, including [Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/)
and an [e-commerce rationale-guided distillation study](https://aclanthology.org/2025.coling-industry.12/),
but neither paper establishes a gain on this dataset. Expected mean Macro gain
should be budgeted as 0 to +0.003 until measured.

Any F1 experiment must generate silver targets for each validation fold from
that fold's outer-training donor pool only. Teacher verdicts, confidence, and
gold-informed rationales must not cross the fold boundary.

### 2. Evidence-grounded silver rationales: highest-value use

This directly addresses the uncovered product requirement. The teacher can
link an exact title/description span or a bounded image region to a closed
concept (`sold_item`, `included_component`, `material`, `fuel`, `ignition`,
`packaging`, `sports_symbolism`, `other`) and then render a 50--300-character
comment consistent with an already locked classifier verdict.

The prompt must ask for evidence on **both sides** without revealing the gold
label or the current model verdict:

```json
{
  "evidence_for_1": [
    {
      "source": "title|description|image",
      "text_span": "exact input substring or null",
      "image_index": 0,
      "bbox_0_1000": [0, 0, 0, 0],
      "observed_text": "OCR text or null",
      "visual_fact": "short visible fact or null",
      "concept": "closed enum value"
    }
  ],
  "evidence_for_0": [],
  "independent_label": 0,
  "confidence": 0.0,
  "ambiguous_or_insufficient": false
}
```

The deterministic renderer later selects evidence matching the locked
classifier verdict for a submission comment, or matching the gold train label
for auxiliary supervision. This design exposes unsupported rationalization:
the teacher cannot simply invent evidence after being told the answer. Exact
text spans are machine-checked; image claims are audited against their region.
ERASER's separation of task output from evidence is the relevant evaluation
principle ([DeYoung et al., 2020](https://aclanthology.org/2020.acl-main.408/));
faithful-by-construction rationalization motivates keeping the renderer tied to
selected evidence ([Jain et al., 2020](https://aclanthology.org/2020.acl-main.409/)).

### 3. OCR and visual attributes: medium value, scoped to hard rows

Qwen3-VL's official model card claims expanded multilingual OCR robust to blur,
tilt, and low light, plus improved fine-grained visual grounding. The official
repository also documents OCR/key-information extraction and object grounding
with points and boxes. These capabilities match card images better than a
text-only teacher, but claims must still be tested locally.

The useful product is not a verbose caption. It is a compact set of localized,
closed attributes: exact visible words, item-vs-packaging role, sold
item-vs-accessory role, material/fuel form, ignition mechanism, and normalized
bounding box. Run this only on the frozen pilot and later on error/low-confidence
rows if the pilot passes. Existing PaddleOCR can serve as a cheap independent
OCR check; previous local work shows that OCR can be valid in strict batch-one
mode but did not improve the strongest classifier when injected broadly.

### 4. Audit of disputed labels: medium diagnostic value, no auto-relabeling

Use the independent teacher label plus dual-sided evidence to prioritize human
review into `supported`, `contradicted`, or `ambiguous/insufficient`. A teacher
can reveal a missing visual cue or a likely ontology mismatch, but cannot
adjudicate an input-indistinguishable mixed-label component. Such families must
be marked irreducibly ambiguous, not "corrected."

No label changes should be made from teacher output alone. A high-confidence
contradiction is merely a queue item for two-person blind review. This protects
the classifier from importing the teacher's commonsense ontology.

## Minimal measurable pilot

### Frozen sample and control

Freeze the row IDs before inference with a deterministic hash tie-break:

- 40 development rows used only to settle prompt, schema, parser, and renderer;
  exclude them from all metrics.
- 200 audit rows: 100 BAD and 100 flammable.
- Within each category, use 50 label-0 and 50 label-1 rows.
- Within each category/label cell of 50: 25 representative current-400 correct,
  15 low-margin/disagreement but correct, and 10 current-400 errors.
- Within BAD, include at least 30 mixed-label connected-component rows. Within
  flammable, cover gas/liquid fuel, ignition/pyrotechnics, solid fuel, and
  included-kit/empty-equipment contrasts. Cohorts may overlap, but every row has
  a recorded selection stratum.
- Require at least 80 audit rows (20 per category/label cell) with human-checkable
  image/OCR evidence.

Run exactly one temperature-zero structured call per row with all available
competition images and the original title/description. Cap generated output at
about 384 tokens. Compare, under the identical prompt/schema:

1. teacher: `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`;
2. small control: `Qwen/Qwen3-VL-2B-Instruct`.

Randomize model labels in the human review UI. Reviewers must not see gold,
current-400 prediction, confidence stratum, or model identity. The classifier's
locked verdict is applied only by the renderer after the blind evidence output
has been saved.

### Hard gates

The pilot passes for explanation generation only if all hard gates pass:

| Gate | Threshold |
|---|---:|
| JSON/schema validity | >=99.5% |
| Locked verdict preserved by renderer | 100% |
| Text evidence is an exact normalized input substring after filtering | 100% |
| Human relevance of rendered comments | >=90% overall and >=85% in every category/label cell |
| Unsupported claims | <1% overall |
| Usable grounded-evidence coverage | >=85% overall and >=80% per category/label cell |
| 50--300 character and output-contract compliance | 100% |
| Repeated 40-row output: same selected source/concept | >=95% |

On the image/OCR subset, additionally require normalized OCR character error
rate <=0.10, claimed region contains the evidence in >=90% of cases, closed
visual-attribute Macro F1 >=0.85, and unsupported visual facts <2%.

Funding a full teacher silver pass additionally requires a measurable advantage
over the 2B control: **at least +8 percentage points** in relevant grounded
comments or **at least +15 points** in usable visual-evidence coverage, without
violating the unsupported-claim gate. If the 2B control already reaches 90%
relevance and the teacher gains <3 points, use the 2B model instead.

For label triage, blind-review 50 of the model's highest-confidence
contradiction/ambiguity flags. Retain the audit tool only if precision for
"genuinely needs human review" is >=80%; never auto-relabel.

### Optional two-fold F1 gate, only after rationale gates pass

Change exactly one factor: add the teacher-derived closed concept/evidence
auxiliary objective to the existing student; preserve its architecture,
classification loss, data, seed, thresholds, and inference. Evaluate frozen
folds 0 and 3 with strictly outer-donor-only silver generation.

Accept only if:

- Macro-F1 delta is positive on both folds;
- mean Macro-F1 delta is >=+0.003;
- neither category drops by more than 0.005;
- flammable/safety false negatives do not increase on either fold; and
- corrected-to-regressed ratio is >=1.5 on the frozen residual audit.

Failure here rejects only the F1-distillation use. Evidence-grounded comments
may still be retained if their own gates pass.

## Model and infrastructure realism

Recommended checkpoint:
[Qwen3-VL-235B-A22B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8).
The [base Instruct card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
documents 236B parameters, 22B active parameters, Apache-2.0 licensing, long
multimodal context, grounding, and OCR capabilities. The
[official Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL) recommends
`transformers>=4.57.0`, vLLM `>=0.11.0`, and for H100/H200 serves this checkpoint
with tensor parallel 8, expert parallel, `mm-encoder-tp-mode=data`, and async
scheduling. The [technical report](https://arxiv.org/abs/2511.21631) is the
primary architectural source.

Use one eight-H100 node following the official TP8 recipe. The FP8 repository is
about 238GB, so the weights fit on one such node and multi-node distribution is
not required. Provider-specific project names, storage paths, credentials and
execution documentation are intentionally excluded from the public repository.
Cache presence has not been verified, so first-use delivery time remains a real
uncertainty.

Use a short 8K--16K serving context; the model's 256K maximum is unnecessary and
would waste KV-cache memory. Before the pilot, a 16-card throughput-only smoke
must measure load time, image preprocessing, tokens/s, peak memory, and parser
validity. It is not an accuracy experiment and may not alter the frozen prompt.

Conservative reservation for the 240-card pilot is 2.5 wall-clock hours on one
`h100-8x` node, or 20 H100-hours, including model delivery/load and both model
runs where practical. This is a planning ceiling, not a measured runtime. Do not
approve a several-thousand-row pass until the 16-card smoke supplies a measured
per-card rate and the 200-row blind gates pass. The organizer reproducibility
requirement also means a full approved generator must be runnable at several
thousand rows, with artifacts saved; a one-off manual/API workflow is invalid.

InternVL3.5-241B-A28B is a plausible research alternative
([official model card](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B)),
but Qwen is preferable for this pilot because the official Qwen project provides
the exact FP8 eight-H100 serving recipe and stronger documented structured OCR
and grounding workflow. This is an operational choice, not a claim that Qwen is
intrinsically more accurate on this dataset.

## Why the small allowed models are sufficient or insufficient

They are sufficient for final inference, deterministic comment rendering,
contract enforcement, exact-span validation, and possibly silver generation if
the 2B control meets the blind gates. They are also the correct deployment
target: the teacher adds no submission-time dependency.

They are not yet demonstrated sufficient for reliable hard-card evidence
discovery. The local 2B/4B visual experiments lost OOF F1, InternVL produced
36.9% invalid structured responses, and the strongest existing renderer has no
card evidence at all. Hard cases require joint reasoning across multiple images,
OCR, sold-item/component role, and a non-intuitive task ontology. The large
teacher's purpose is to test whether extra capacity resolves that evidence
bottleneck. If it fails to exceed the 2B control by the predeclared margin, its
cost is unjustified and the small model is sufficient.

## Stop conditions

Stop after the 40-row development phase if schema validity is below 98% after
one parser-only correction, or if unsupported evidence exceeds 5%. Stop after
the blind 200-row audit if any hard gate fails. Stop the F1 branch after either
fold fails its gate. Do not expand generation merely because examples look
persuasive, and do not use teacher disagreement to rewrite ground truth.

The resulting decision is deliberately asymmetric: a large teacher can be
valuable without moving F1, because it can make final comments concrete and
auditable. It earns a full-dataset run only by demonstrating a material grounded
evidence advantage over the allowed 2B control on our own frozen cards.
