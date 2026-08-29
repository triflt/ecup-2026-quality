# Architecture of solution 140

```text
name + description ──► robust text classifier ───────────────┐
images + text ───────► Qwen3-VL-2B + rsLoRA ────────────────┼─► category fusion
images + text ───────► Qwen3.5-4B + rsLoRA ────────────────┤
train-only exact/name donors ─► product-memory prior ───────┘
                                                        │
                                                        └─► verdict + comment
```

The base models are loaded from `SHARED_MODELS_PATH`. The two small LoRA
adapters and fitted classifier bundles are submission artifacts. Qwen3-VL uses
the same 448-pixel preprocessing that passed the corrected official-image
smoke. Qwen3.5 and Qwen3-VL produce independent probabilities; fusion is
category-specific because the two competition categories have different error
profiles.

Exact/name memory is a donor-only feature, not a test-label lookup. The offline
validation code fits it only from the training side of each split. The final
runner uses the memory serialized from the full competition training set.

The current explanation layer is format-safe but is being replaced by a
separately audited evidence-grounded layer. That work must not change the
classification verdict unless it is evaluated as a new solution.
