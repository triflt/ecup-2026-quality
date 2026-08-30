from __future__ import annotations

from pathlib import Path

EXPERIMENT_ID = "601"
VALIDATION_VERSION = "semantic_family_v3"
ROBUST_PROTOCOL_VERSION = "semantic_v3_robust_base_strict_nested_v2"
DEVELOPMENT_FOLDS = (0, 1, 2, 3, 4)
SEALED_FOLD = -1
QWEN_SEED = 42
QWEN_FIRST_IMAGE_MAX_EDGE = 448
QWEN_FIRST_IMAGE_MAX_PIXELS = 262_144
EXPECTED_DEVELOPMENT_ROWS = 11_118
EXPECTED_SEALED_ROWS = 1_853
EXPECTED_FOLDS_SHA256 = "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae"
EXPECTED_QWEN_PARENT_SHA256 = "c30e690ad260af72fcc625c8d3e6d9ab9c5a096d8443d6d9f5f7adbcaa52123c"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "research/data.csv"
DEFAULT_FOLDS = ROOT / "validation/semantic_family_v3/folds.csv"
DEFAULT_FIRST_MANIFEST = ROOT / "research/lora_image_manifest_complete.tsv.gz"
DEFAULT_ALL_EMBEDDINGS = ROOT / "research/all-image-artifacts/extracted/train_embeddings_fp16.npz"
DEFAULT_FIRST_EMBEDDINGS = (
    ROOT / "research/first-image-artifacts/extracted/train_embeddings_fp16.npz"
)


def require_development_fold(fold: int) -> int:
    if fold not in DEVELOPMENT_FOLDS:
        raise ValueError(f"fold {fold} is not one of {DEVELOPMENT_FOLDS}")
    return fold
