from __future__ import annotations

import importlib
import sys
from collections.abc import MutableMapping
from pathlib import Path
from types import ModuleType

from contract import PARENT_ENVIRONMENT, PARENT_RECIPE_SHA256, SCREEN_FOLDS, SEED

ROOT = Path(__file__).resolve().parents[2]
PARENT_MODULE = "research.qwen35_bad_family_diverse_positives_lora"
PARENT_PATH = ROOT / "research/qwen35_bad_family_diverse_positives_lora.py"


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_parent_environment(
    *, fold: int, output_dir: Path, environment: MutableMapping[str, str]
) -> None:
    if fold not in SCREEN_FOLDS:
        raise ValueError(f"fold must be one of the predeclared screen folds {SCREEN_FOLDS}")
    locked = {
        **PARENT_ENVIRONMENT,
        "SEED": str(SEED),
        "HOLDOUT_FOLD": str(fold),
        "FULL_TRAIN": "0",
        "ECUP_OUTPUT_DIR": str(output_dir),
    }
    for key, expected in locked.items():
        actual = environment.get(key)
        if actual not in (None, "", expected):
            raise ValueError(f"parent environment {key} must remain {expected!r}")
        environment[key] = expected
    forbidden = {
        "LINEAR_ONLY_TARGETS": {"1", "true"},
        "SOFT_TARGETS": {"1", "true"},
        "DOWNSAMPLE_MODE": {"1", "true"},
    }
    enabled = [
        key
        for key, values in forbidden.items()
        if environment.get(key, "").strip().lower() in values
    ]
    if enabled:
        raise ValueError(f"non-parent switches are forbidden: {sorted(enabled)}")


def load_parent_module() -> ModuleType:
    if sha256(PARENT_PATH) != PARENT_RECIPE_SHA256:
        raise ValueError("frozen parent recipe checksum mismatch")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parent = importlib.import_module(PARENT_MODULE)
    expected = {
        "SEED": SEED,
        "TRAINING_MODE": "hard",
        "MODEL_CLASS": "multimodal",
        "USE_CHAT_BATCH": True,
        "FULL_TRAIN": False,
        "FAMILY_BALANCE_FLAMMABLE": False,
        "FAMILY_DIVERSE_BAD_POSITIVES": True,
        "FAMILY_DIVERSE_FLAMMABLE_NEGATIVES": False,
        "DESCRIPTION_LIMIT": 1800,
        "EPOCHS": 1,
        "BATCH_SIZE": 4,
        "GRAD_ACCUM": 4,
    }
    mismatch = {
        key: {"expected": value, "actual": getattr(parent, key, None)}
        for key, value in expected.items()
        if getattr(parent, key, None) != value
    }
    if mismatch or parent.HOLDOUT_FOLD not in SCREEN_FOLDS:
        raise ValueError(f"parent recipe mismatch: {mismatch}")
    return parent
