# Research entrypoints

The Python files in this directory are stable historical entrypoints used by
experiment packages. They intentionally remain flat: moving them would break
recorded commands and provenance links. There are no byte-identical duplicate
scripts in this directory.

## Solution 140 path

- `train_full_fusion.py` — robust text and embedding base training.
- `aggregate_lora_oof.py` — fold prediction aggregation.
- `nested_lora_calibration.py` — leakage-safe calibration of one LoRA branch.
- `nested_multimodel_fusion.py` — nested fusion of the robust base, Qwen3-VL
  LoRA and Qwen3.5-4B LoRA.
- `dual_lora_submission_smoke_bootstrap.py` and
  `dual_lora_runtime_smoke_bootstrap.py` — submission and scaled-runtime checks.

The canonical commands and artifact contract are documented in
[`experiments/140_dual_lora_fusion/final/`](../experiments/140_dual_lora_fusion/final/).

## Other groups

- `*_cv.py`, `*_audit.py`: offline comparisons and fail-closed audits.
- `qwen35_*`, `qwen3vl_*`: model-specific training and OOF utilities.
- `paddleocr_*`: OCR diagnostics; OCR is not part of the current solution 140
  runtime.
- `*_bootstrap.py`: reproducible environment or runtime smoke builders.

New reusable code should go under `src/ecup_quality/`. New experiment-specific
code should live in its experiment directory. Do not add more one-off scripts
to this directory unless an existing immutable command depends on that path.
