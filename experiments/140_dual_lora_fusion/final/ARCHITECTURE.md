# Architecture of solution 140

| Component | Text input | Visual input | Role |
|---|---|---|---|
| TF-IDF LinearSVC | name + description | none | stable text rank |
| Qwen3-VL-Embedding-2B | name + description + category | **all available images** | multimodal rank inside the robust base |
| Qwen3-VL-2B rsLoRA | name + description + category | **first image, max 448 px** | supervised multimodal score |
| Qwen3.5-4B rsLoRA | name + description + category | **the same first image, max 448 px** | supervised multimodal reasoning score |
| Product memory | normalized name/text | none | train-only exact/name correction after fusion |

The TF-IDF and embedding ranks first form the robust base. Category-specific
fusion then combines that base with both LoRA scores; product memory is applied
after fusion. Thus all three Qwen components receive visual input. The embedding
model consumes every available image, while the two LoRA branches deliberately
share the same first-image view. If the embedding model hits a single-sample
CUDA OOM, that one row has an explicit text-only fallback; the fallback is
logged rather than silently changing the entire route.

The base models are loaded from `SHARED_MODELS_PATH`. The two small LoRA
adapters and fitted classifier bundles are submission artifacts. Both LoRA
branches use the same 448-pixel first-image preprocessing that passed the
corrected official-image smoke. Qwen3.5 and Qwen3-VL produce independent probabilities; fusion is
category-specific because the two competition categories have different error
profiles.

Exact/name memory is a donor-only feature, not a test-label lookup. The offline
validation code fits it only from the training side of each split. The final
runner uses the memory serialized from the full competition training set.

The current explanation layer is format-safe but is being replaced by a
separately audited evidence-grounded layer. That work must not change the
classification verdict unless it is evaluated as a new solution.
