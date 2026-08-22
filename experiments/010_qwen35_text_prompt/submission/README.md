# Qwen3.5-4B prompt submission

The evaluator mounts `Qwen/Qwen3.5-4B` under `SHARED_MODELS_PATH`. The archive
contains only inference code and the prompt; model weights are not bundled.

```bash
python -u run.py --test_data_path /data/test.csv --output_path /output/submission.csv
```

Optional environment overrides: `QWEN_MODEL_PATH`, `QWEN_BATCH_SIZE`,
`QWEN_MAX_INPUT_TOKENS`, and `QWEN_MAX_NEW_TOKENS`.
