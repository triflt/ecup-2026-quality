# Experiment 621: semantic-v3 Fisher blockwise merge

Status: **rejected before validation at the frozen reconstruction gate**.

This package defines a public-safe, deterministic way to merge two adapters
trained from the same base model and semantic-v3 outer fold. The original
adapter supplies the broad behavior. A specialist adapter contributes only
through conservative per-module coefficients computed from Fisher statistics
calculated on the corresponding training partition. No validation predictions,
historical OOF matrices, sealed rows, or large teacher are part of the merge.

The implementation operates on effective LoRA updates (`scale * B @ A`), never
on A and B independently. The merged update is compressed back to the declared
rank with deterministic SVD signs. Reconstruction error and metadata identity
must pass before a run can proceed.

The intended grid was exactly five one-GPU jobs, one per frozen outer fold. Both
predeclared screen folds stopped before validation scoring because the same
layer-11 `k_proj` update exceeded the frozen `0.05` relative SVD reconstruction
limit: `0.0944161222` on fold 0 and `0.0956405199` on fold 3. The limit was not
relaxed after observation. No validation result is claimed; folds 1/2/4 and a
full-data refit are forbidden.

The label-blind preflight accepts only the safe membership columns:
`id`, `category`, `semantic_component`, `component_size`, `split`, and
`development_fold`. It rejects supervision columns and private infrastructure
references in publication metadata.

## Runtime builder and GPU worker

`build_fold_runtime.py` takes the source development table, the frozen
semantic-v3 membership, and an image manifest. It writes four explicit files:

- `train_data.csv`, containing labels for the four training folds;
- `validation_data.csv`, containing the requested fold with no `label` column;
- `development_data_label_free.csv` and `development_membership.csv`;
- `development_image_manifest.tsv`, filtered to development IDs only.

The builder writes hashes and a self-audit proving that validation labels and
sealed rows were not written. The worker receives this runtime directory rather
than the source table.

`run_fold.py` safely unpacks a pair of PEFT adapter directories or archives,
selects deterministic one-per-component Fisher samples, loads one
adapter at a time, and computes mean squared LoRA gradients for the broad and
specialist tasks. It then calls the existing effective-delta merge, loads the
merged adapter, and writes label-free validation scores plus a hash-bound fold
contract. Its prompt, text normalization, no-thinking chat encoding, 1536-token
limit, multimodal model loader, and parent source hashes are frozen to the two
experiment-600 parents. Both exact parent runner files are mandatory runtime
inputs and are checksum-verified before any model or image work. The first-image
policy is strict: every required image is downloaded from the development-only
runtime manifest, decoded, resized, and materialized under a fresh
`--image-root`; any failure aborts the fold and there is no synthetic image
fallback.

## Local fold workflow

`fold_worker.py` accepts extracted PEFT adapter directories containing
`adapter_config.json` and `adapter_model.safetensors`. It checks the base-model
identity, target modules, rank, scaling convention, adapter tensor hashes, and
semantic-v3 membership. A Fisher report must explicitly carry the train-ID hash,
zero validation/sealed rows, and `validation_labels_loaded=false`. The worker
then writes a new adapter plus `merge_manifest.json`; it does not load labels.

```bash
python experiments/621_semantic_v3_fisher_blockwise_merge/fold_worker.py \
  --data /path/to/development.csv \
  --folds /path/to/semantic_v3_folds.csv \
  --outer-fold 0 \
  --original-adapter /path/to/original_adapter \
  --specialist-adapter /path/to/specialist_adapter \
  --fisher-report /path/to/train_only_fisher.json \
  --base-model-id public/model \
  --base-model-revision revision \
  --output-dir /path/to/merged_adapter
```

The Fisher estimation interface is `estimate_fisher_from_gradients`: a model
runtime supplies per-module gradients from the train partition, and the helper
returns squared-gradient means with explicit zero counts for validation and
sealed rows. No model weights or data are bundled here.

After an external runtime scores the merged adapter, `evaluate_fold.py` joins a
prediction file (`id,category,fold,lora_score`) to the requested validation
fold and applies pre-frozen thresholds. It rejects prediction files that carry
labels, rejects sealed input rows, and never tunes thresholds on validation.

Expected resources are one GPU with approximately 24–40 GB of memory, the
local base model and PEFT runtime, and roughly 4 GB of temporary adapter/model
working space. The two screen jobs reached the deterministic merge gate after
their train-only Fisher passes, then failed closed before label-free validation
scoring. This is a scientific rejection of the fixed-rank merge, not a transport
failure and not grounds for changing the rank or error limit under experiment 621.
