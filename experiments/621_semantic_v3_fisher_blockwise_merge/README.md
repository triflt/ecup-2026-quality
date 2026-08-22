# Experiment 621: semantic-v3 Fisher blockwise merge

Status: **runnable_not_launched; not validated**.

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

The intended grid is exactly five one-GPU jobs, one per frozen outer fold. This
repository change does not launch jobs, access model storage, download data, or
claim any validation result. A full-data refit is forbidden until the five-fold
acceptance gates in `experiment.toml` pass.

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
selects deterministic one-per-component-per-label Fisher samples, loads one
adapter at a time, and computes mean squared LoRA gradients for the broad and
specialist tasks. It then calls the existing effective-delta merge, loads the
merged adapter, and writes label-free validation scores plus a hash-bound fold
contract. The first-image policy is strict: images must already be available
under `--image-root`; the worker never downloads them.

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
working space. A fold is expected to take approximately 50–90 minutes, mainly
for two gradient passes and label-free validation scoring; this is an estimate,
not an observed result. The current blockers are model/runtime availability,
complete local first-image assets, and a valid train-only Fisher report path.
