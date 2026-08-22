from __future__ import annotations

"""Locked launcher for the two-fold resolution screen.

The actual model, selector, prompt, optimization and evaluation remain in the
experiment-110 parent runner. This module only freezes the one permitted image
resolution change and refuses full training or unregistered folds.
"""

import argparse
import os
import runpy
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path

from resolution_contract import (
    CANDIDATE_MAX_EDGE,
    CANDIDATE_MAX_PIXELS,
    SCREEN_FOLDS,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT_RUNNER = ROOT / "research/qwen3vl_lora_holdout.py"
LOCKED_ENVIRONMENT = {
    "SEED": "42",
    "TRAINING_MODE": "hard",
    "MODEL_CLASS": "image_text",
    "DESCRIPTION_LIMIT": "1800",
}
LOCKED_DISABLED_ENVIRONMENT = {
    "FULL_TRAIN": ("", "0", "false"),
    "SOFT_TARGETS": ("",),
    "DOWNSAMPLE_MODE": ("",),
    "MAX_SLICE_NUMS": ("", "0"),
    "LAST_LOGIT_ONLY": ("", "0", "false"),
    "USE_CHAT_BATCH": ("", "0", "false"),
    "LINEAR_ONLY_TARGETS": ("", "0", "false"),
}
RESOLUTION_ENVIRONMENT = {
    "QWEN3VL_FIRST_IMAGE_MAX_EDGE": str(CANDIDATE_MAX_EDGE),
    "QWEN3VL_FIRST_IMAGE_MAX_PIXELS": str(CANDIDATE_MAX_PIXELS),
}


def configure_environment(
    fold: int, environment: MutableMapping[str, str]
) -> MutableMapping[str, str]:
    if fold not in SCREEN_FOLDS:
        raise ValueError(f"fold {fold} is not in the predeclared screen {SCREEN_FOLDS}")
    existing_fold = environment.get("HOLDOUT_FOLD")
    if existing_fold not in (None, "", str(fold)):
        raise ValueError("HOLDOUT_FOLD conflicts with the explicit locked fold")
    for key, expected in LOCKED_ENVIRONMENT.items():
        current = environment.get(key)
        if current not in (None, "", expected):
            raise ValueError(f"{key} must remain at the parent value {expected}")
        environment[key] = expected
    enabled = [
        key
        for key, disabled_values in LOCKED_DISABLED_ENVIRONMENT.items()
        if environment.get(key, "").strip().lower() not in disabled_values
    ]
    if enabled:
        raise ValueError(f"non-parent switches are forbidden: {sorted(enabled)}")
    for key, expected in RESOLUTION_ENVIRONMENT.items():
        current = environment.get(key)
        if current not in (None, "", expected):
            raise ValueError(f"{key} conflicts with the locked resolution")
        environment[key] = expected
    environment["HOLDOUT_FOLD"] = str(fold)
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=SCREEN_FOLDS)
    args = parser.parse_args(argv)
    configure_environment(args.fold, os.environ)
    if not PARENT_RUNNER.is_file():
        raise FileNotFoundError(PARENT_RUNNER)
    sys.argv = [str(PARENT_RUNNER)]
    runpy.run_path(str(PARENT_RUNNER), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
