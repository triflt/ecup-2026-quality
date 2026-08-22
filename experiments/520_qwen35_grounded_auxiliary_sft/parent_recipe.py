from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
PARENT_MODULE = "research.qwen35_bad_family_diverse_positives_lora"

FIXED_PARENT_ENV = {
    "TRAINING_MODE": "hard",
    "MODEL_CLASS": "multimodal",
    "USE_CHAT_BATCH": "1",
    "FAMILY_BALANCE_FLAMMABLE": "0",
    "FAMILY_DIVERSE_BAD_POSITIVES": "1",
    "FAMILY_DIVERSE_FLAMMABLE_NEGATIVES": "0",
    "DESCRIPTION_LIMIT": "1800",
}


def configure_parent_environment() -> None:
    for key, expected in FIXED_PARENT_ENV.items():
        actual = os.environ.setdefault(key, expected)
        if actual != expected:
            raise ValueError(f"experiment 520 freezes {key}={expected!r}; received {actual!r}")


def load_parent_module() -> ModuleType:
    configure_parent_environment()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parent = importlib.import_module(PARENT_MODULE)
    assert_exact_parent(parent)
    return parent


def assert_exact_parent(parent: ModuleType) -> None:
    expected = {
        "TRAINING_MODE": "hard",
        "MODEL_CLASS": "multimodal",
        "USE_CHAT_BATCH": True,
        "FAMILY_BALANCE_FLAMMABLE": False,
        "FAMILY_DIVERSE_BAD_POSITIVES": True,
        "FAMILY_DIVERSE_FLAMMABLE_NEGATIVES": False,
        "DESCRIPTION_LIMIT": 1800,
        "EPOCHS": 1,
        "BATCH_SIZE": 4,
        "GRAD_ACCUM": 4,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(parent, key, None)}
        for key, value in expected.items()
        if getattr(parent, key, None) != value
    }
    if mismatches:
        raise ValueError(f"parent recipe mismatch: {mismatches}")
