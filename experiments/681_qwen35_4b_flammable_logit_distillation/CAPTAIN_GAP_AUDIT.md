# Top-level competition gap audit

This board separates system capabilities from individual local hypotheses.

| Winning-solution pillar | Current state | Gap | Next evidence |
|---|---|---|---|
| Leakage-safe validation | Strong frozen semantic folds, SHA-bound replay and category slices | Public shift remains materially worse than local replay | Keep family/singleton gates and an independent final candidate |
| Rare-class ranking | Canonical tie-aware AP is now measured; 27B probe improved all five folds | Deployable 4B has not retained the ranking signal; checkpoint selection by inner AP is not yet proven | 681 fixed logit distillation, then nested AP checkpointing only if needed |
| Category specialization | Separate routes are deployable and BAD can remain unchanged | Two direct flammable 4B training variants were unstable | Transfer a proven orthogonal signal instead of another class-only restart |
| Images and decisive text | Multimodal base/LoRAs exist; large offline OCR corpus is preserved fail-closed | Critical-span recall is unproven and ordinary OCR injection failed | Use OCR for auditable evidence first; no raw-text injection |
| Retrieval / new families | Known-repeat handling is already very strong; external embedder work shows retrieval gains | Singleton semantic families dominate errors; new embedder is not integrated/calibrated | One isolated embedding+classifier refit with unchanged downstream fusion |
| Label noise | Mixed-label cohorts are identified | Robust loss has not passed a clean isolated test; upside is limited | Defer conflict-aware loss until primary lanes close |
| Diverse ensemble | Offline 27B errors are complementary | 27B is undeployable; no accepted full-data diverse student exists | Distill signal into 4B; keep evidence/retrieval candidate independent |
| Hyperparameter dynamics | Recipe and runtime are reproducible; LR ablation was tested | Final checkpoint optimality is not proven by inner AP | Only leakage-safe inner checkpoint screen; do not use outer folds |
| Inference/runtime | One base with category adapters and one forward per row is feasible | Any extra pass threatens Private runtime | Require mixed-category official smoke and runtime projection |
| Submission discipline | Builders are fail-closed and artifacts are SHA-bound | No new accepted full-data adapter exists yet | Package automatically immediately after full or fallback gate |

## Ranked lanes

1. **681 distillation** — highest evidence because the teacher correction signal
   is 5/5-fold positive and directly targets the rare class. Cost after targets:
   two-GPU screen, then three-GPU confirmation.
2. **Embedding/retrieval ablation** — independent error type and promising
   external result, but requires refitting the compatible multimodal classifier
   and recalibrating on training folds; swapping only the encoder is invalid.
3. **Evidence/span candidate** — potential BAD/singleton gain, but currently lacks
   a full-data custom-head refit and deployable package.
4. **Nested AP checkpointing** — useful only if 681 is directionally positive but
   training dynamics are unstable. It must use inner folds and a frozen mapping
   to exact optimizer steps.

## Explicit stops

- No 27B model or adapter in a submission.
- No rejected 4B screen adapter in a submission.
- No threshold or prevalence matching based on Public/Private class balance.
- No naive flammable keyword rules; frozen replay showed large regressions.
- No raw OCR as ordinary text and no third OCR repair architecture.
- No encoder-only replacement while retaining a classifier trained in the old
  embedding space.
