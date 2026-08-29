# Reproducing solution 140

The code path is reproducible from a clean checkout. Exact historical numeric
reproduction additionally requires the private training data, pretrained model
revisions, two LoRA adapters and three frozen OOF arrays whose identities are
listed below. The binary weights will be published separately.

## 1. Environment and immutable inputs

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,vlm]'

python tools/audit_dataset.py --data /path/to/data.csv --images /path/to/images
```

Use `competition_train_v1` and the checked-in
`validation/grouped_text_v1/folds.csv`. Do not regenerate the folds in place.

## 2. Train Qwen3-VL-2B rsLoRA folds

```bash
for fold in 0 1 2 3 4; do
  python experiments/110_qwen3vl_lora/run.py \
    --data /path/to/data.csv \
    --images /path/to/images \
    --model-root /path/to/Qwen3-VL-2B-Instruct \
    --output-dir /tmp/solution140/qwen3vl/fold${fold} \
    --set ECUP_MANIFEST=/path/to/lora_image_manifest.tsv.gz \
    --set ECUP_OOF=/path/to/robust_base_oof.npz \
    --set HOLDOUT_FOLD=${fold} \
    --set SEED=42 \
    --set TRAINING_MODE=hard \
    --set MODEL_CLASS=image_text
done
```

## 3. Train Qwen3.5-4B rsLoRA folds

```bash
for fold in 0 1 2 3 4; do
  python experiments/130_qwen35_lora/run.py \
    --data /path/to/data.csv \
    --images /path/to/images \
    --model-root /path/to/Qwen3.5-4B \
    --output-dir /tmp/solution140/qwen35/fold${fold} \
    --set ECUP_MANIFEST=/path/to/lora_image_manifest.tsv.gz \
    --set ECUP_OOF=/path/to/robust_base_oof.npz \
    --set HOLDOUT_FOLD=${fold} \
    --set SEED=42 \
    --set TRAINING_MODE=hard \
    --set MODEL_CLASS=multimodal \
    --set USE_CHAT_BATCH=1
done
```

The shared runner freezes one epoch, maximum length 1536, micro-batch 4,
gradient accumulation 4, seed 42, rsLoRA rank 16 / alpha 32 / dropout 0.05,
and first-image edge 448.

For the two full-data adapters, repeat the corresponding command with a new
output directory and `--set FULL_TRAIN=1`.

## 4. Aggregate each model's five OOF prediction files

```bash
python research/aggregate_lora_oof.py \
  --predictions /tmp/solution140/qwen3vl/fold{0,1,2,3,4}/predictions.csv \
  --base-oof /path/to/robust_base_oof.npz \
  --output /tmp/solution140/qwen3vl_aggregate.json

python research/aggregate_lora_oof.py \
  --predictions /tmp/solution140/qwen35/fold{0,1,2,3,4}/predictions.csv \
  --base-oof /path/to/robust_base_oof.npz \
  --output /tmp/solution140/qwen35_aggregate.json
```

Each command also writes a companion `.npz` used by the fusion step.

## 5. Reproduce nested fusion

```bash
python experiments/140_dual_lora_fusion/run.py \
  --inputs /tmp/solution140/qwen3vl_aggregate.npz \
           /tmp/solution140/qwen35_aggregate.npz \
  --names qwen3vl qwen35 \
  --step 0.05 \
  --output /tmp/solution140/nested_fusion_report.json
```

`--step 0.05` is required to represent the frozen weights exactly:

- БАД: robust base / Qwen3-VL / Qwen3.5 = `0.50 / 0.25 / 0.25`;
- Легковоспламеняющиеся: `0.15 / 0.10 / 0.75`.

## Frozen historical OOF identities

| Object | SHA-256 |
|---|---|
| Fold assignments | `03baaa25bd5a3aef6ad94e02067cccda114041f98d7a35a9e06330a425166e4d` |
| Robust/base OOF | `5d7467c48fc8a5a73f947f5aa1300071c77ba699b29c12250caf1bcd3176d7ac` |
| Qwen3-VL aggregate OOF | `ba432e13624e6c3b1c7304ced8cacf580f4ffcc0a0cde1af8b6afb098bf6dc01` |
| Qwen3.5 seed-42 aggregate OOF | `147f2b2b87d8220566526b38ab0e085c0bc44cd82b0442baca6b019516c3f1d8` |

The last three arrays are not committed. Publishing them or immutable download
references closes exact historical score reproduction; until then the source
recipe is reproducible but the original cached-result replay is intentionally
marked incomplete.
