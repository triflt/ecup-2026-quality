from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXED_ENV = {
    "TRAINING_MODE": "hard",
    "MODEL_CLASS": "multimodal",
    "USE_CHAT_BATCH": "1",
    "FAMILY_BALANCE_FLAMMABLE": "0",
    "FAMILY_DIVERSE_BAD_POSITIVES": "1",
    "FAMILY_DIVERSE_FLAMMABLE_NEGATIVES": "0",
    "DESCRIPTION_LIMIT": "1800",
}


def load_parent_module():
    for key, expected in FIXED_ENV.items():
        actual = os.environ.setdefault(key, expected)
        if actual != expected:
            raise ValueError(f"experiment 560 freezes {key}={expected!r}; received {actual!r}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parent = importlib.import_module("research.qwen35_bad_family_diverse_positives_lora")
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
    mismatch = {
        key: {"expected": value, "actual": getattr(parent, key, None)}
        for key, value in expected.items()
        if getattr(parent, key, None) != value
    }
    if mismatch:
        raise ValueError(f"parent recipe mismatch: {mismatch}")
    if parent.FULL_TRAIN or parent.HOLDOUT_FOLD not in (0, 3) or parent.SEED != 42:
        raise ValueError("experiment 560 permits only seed-42 outer folds 0 and 3")
    return parent
