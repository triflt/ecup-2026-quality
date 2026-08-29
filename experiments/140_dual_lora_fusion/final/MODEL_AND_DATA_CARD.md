# Model and data card

## Data

Solution 140 is trained only on the provided E-CUP 2026 Quality training set:
`competition_train_v1`, 12,971 rows. Its immutable identity is registered in
[`datasets/registry.toml`](../../../datasets/registry.toml). Raw rows and images
are intentionally not committed.

No external product dataset and no synthetic training corpus is part of the
submitted solution-140 recipe.

## Models

| Component | Model | License status |
|---|---|---|
| Multimodal embedding | `Qwen/Qwen3-VL-Embedding-2B` | Competition-listed, Apache-2.0 |
| Multimodal LoRA | `Qwen/Qwen3-VL-2B-Instruct` | Competition-listed, Apache-2.0 |
| Reasoning LoRA | `Qwen/Qwen3.5-4B` | Competition-listed, Apache-2.0 |
| Text classifiers | scikit-learn TF-IDF / linear models | Source and pinned dependency are public |

The base weights are provided by the competition runtime and are not copied
into this repository. The two trained adapters will be published separately
with exact SHA-256 values.
