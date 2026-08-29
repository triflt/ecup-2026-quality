# Solution 140 — finalist handoff

This directory is the single review entrypoint for **solution 140**, the current
submitted production champion. Existing experiment-140 sources remain in their
original locations; this handoff links and verifies them without duplicating
model code.

## What solution 140 is

Solution 140 combines three learned signals and a train-only memory layer:

1. a robust TF-IDF / embedding classifier base;
2. a Qwen3-VL-2B LoRA branch for multimodal evidence;
3. a Qwen3.5-4B LoRA branch for category reasoning;
4. exact and normalized-name product memory fitted only from available training
   donors.

The category-specific fusion weights are selected inside the four development
folds and applied to the held-out outer fold. The submitted archive obtained
Public Macro F1 `0.8923976821312729`.

## Review map

| Question | Document or source |
|---|---|
| Architecture | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Training and reproduction | [`REPRODUCE.md`](REPRODUCE.md) |
| Validation and local leaderboard | [`EVALUATION.md`](EVALUATION.md) |
| Offline/runtime/size meaning | [`RUNTIME_AND_SIZE.md`](RUNTIME_AND_SIZE.md) |
| Models, licenses and data | [`MODEL_AND_DATA_CARD.md`](MODEL_AND_DATA_CARD.md) |
| Weights and immutable artifacts | [`artifact-contract.json`](artifact-contract.json) |
| Repository-level verifier | [`verify.py`](verify.py) |
| Official inference entrypoint | [`../submission/run.py`](../submission/run.py) |
| Official metadata | [`../submission/metadata.json`](../submission/metadata.json) |
| Recorded metrics | [`../results/metrics.json`](../results/metrics.json) |

## Quick verification without weights

```bash
python3 experiments/140_dual_lora_fusion/final/verify.py
pytest -q tests/test_solution_140_repository.py
```

The verifier checks the repository contract, model/data identities, official
CLI, output schema guard and consistency of the recorded Public score. Weight
checks become active after the files and SHA-256 values are published in the
artifact contract.

## Build the submission after weights are published

```bash
python experiments/140_dual_lora_fusion/build_submission.py \
  --output /tmp/solution-140.zip
sha256sum /tmp/solution-140.zip
```

The expected SHA of the already submitted immutable archive is recorded in
[`reports/champion.json`](../../../reports/champion.json). A rebuilt archive is
accepted only when its own manifest, schema smoke and runtime replay pass; it
must not silently inherit the historical SHA.
