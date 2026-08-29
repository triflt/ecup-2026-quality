# Repository readiness for solution 140

This is an evidence-based readiness audit, not an invented jury score. The
organizers evaluate the executable solution, Public/Private Macro F1, compliance
with the rules and the quality of explanations. Repository quality matters
because it must make those claims reproducible and auditable.

The source requirements are summarized in
[`task-and-rules.md`](task-and-rules.md).

| Criterion | Status | Evidence | Remaining action |
|---|---|---|---|
| Official `-i/-o` launch | PASS | [`submission/run.py`](../../experiments/140_dual_lora_fusion/submission/run.py), repository verifier | Repeat in final immutable archive |
| Output CSV and tag schema | PASS | formatter guard and [`test_solution_140_repository.py`](../../tests/test_solution_140_repository.py) | Repeat schema smoke with published weights |
| Offline inference | PASS (source path) | top-level import audit and local-only image URI contract in [`verify.py`](../../experiments/140_dual_lora_fusion/final/verify.py) | Container-level network-disabled smoke |
| Macro F1 evidence | PASS | [`EVALUATION.md`](../../experiments/140_dual_lora_fusion/final/EVALUATION.md), [`champion.json`](../../reports/champion.json) | Do not mix historical nested and semantic-v3 rows |
| Public result provenance | PASS | immutable archive SHA and Public `0.8923976821` in [`champion.json`](../../reports/champion.json) | Private score remains unknown |
| Explanation specificity | BLOCKED | explicit `final_readiness.explanations=false` | Finish the independent explanation gate |
| Runtime limits | PASS (historical) | 600-row smoke: 228.76 s; projected 10.17/24.15 min | Replay unchanged final archive |
| Archive/image size | PENDING | limits and required artifacts documented | Record final ZIP and image sizes after weights are published |
| Training reproducibility | PASS (code path) | [`REPRODUCE.md`](../../experiments/140_dual_lora_fusion/final/REPRODUCE.md) | Publish immutable OOF/artifact references |
| Exact historical replay | PENDING | frozen expected SHAs are recorded | Publish the three OOF arrays or immutable references |
| Model/data provenance | PASS (metadata) | [`MODEL_AND_DATA_CARD.md`](../../experiments/140_dual_lora_fusion/final/MODEL_AND_DATA_CARD.md), registries | Add final model/processor revision hashes with weights |
| No forbidden external data | PASS (declared) | [`datasets/registry.toml`](../../datasets/registry.toml) and dataset audit | Re-run audit on final train manifest |
| Synthetic-data compliance | NOT USED BY SOLUTION 140 | solution 140 does not depend on a synthetic branch | If a synth branch replaces a component, publish generator/model/license/method |

## Honest conclusion

The classification repository for solution 140 is reviewable and its code path
is reproducible. It is not yet a fully closed finalist artifact because binary
weights/OOF references, a fresh immutable runtime-size replay and the explanation
gate are still outstanding. Those gaps are explicit and machine-checkable; no
historical local score is presented as a substitute for them.
