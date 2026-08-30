from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol import DEVELOPMENT_FOLDS, canonical_sha256


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ArrayArchive:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self._values = values
        self.files = list(values)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._values[key]


def _arrays(source) -> ArrayArchive:
    return ArrayArchive({key: source[key].copy() for key in source.files})


def run_old_protocol_parity(data_path: Path, oof_path: Path) -> dict:
    environment = dict(os.environ)
    environment.update(
        {
            "SEED": "42",
            "FULL_TRAIN": "0",
            "TRAINING_MODE": "hard",
            "MODEL_CLASS": "multimodal",
            "USE_CHAT_BATCH": "1",
            "DESCRIPTION_LIMIT": "1800",
            "FAMILY_BALANCE_FLAMMABLE": "0",
            "FAMILY_DIVERSE_BAD_POSITIVES": "1",
            "FAMILY_DIVERSE_FLAMMABLE_NEGATIVES": "0",
        }
    )
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    try:
        original = _load("_exp600_parity_original", ROOT / "research/qwen3vl_lora_holdout.py")
        specialist = _load(
            "_exp600_parity_specialist",
            ROOT / "research/qwen35_bad_family_diverse_positives_lora.py",
        )
    finally:
        os.environ.clear()
        os.environ.update(previous)
    frame = pd.read_csv(data_path, dtype={"id": str})
    source = np.load(oof_path, allow_pickle=True)
    if not np.array_equal(frame["id"].astype(str), source["ids"].astype(str)):
        raise ValueError("old-protocol parity input ids differ")
    projected_frame = frame.reset_index(drop=True).copy()
    projected_oof = _arrays(source)
    fold_reports = []
    for fold in DEVELOPMENT_FOLDS:
        original.HOLDOUT_FOLD = fold
        specialist.HOLDOUT_FOLD = fold
        control_records = original.select_training(frame, source)
        projected_control = original.select_training(projected_frame, projected_oof)
        specialist_records, _ = specialist.select_training(frame, source)
        projected_specialist, _ = specialist.select_training(projected_frame, projected_oof)
        fold_reports.append(
            {
                "fold": fold,
                "original_exact": control_records == projected_control,
                "original_records": len(control_records),
                "original_records_sha256": canonical_sha256(control_records),
                "specialist_exact": specialist_records == projected_specialist,
                "specialist_records": len(specialist_records),
                "specialist_records_sha256": canonical_sha256(specialist_records),
            }
        )
    return {
        "audit_version": "semantic_v3_qwen35_selector_null_parity_v1",
        "mode": "old_protocol_identity_projection",
        "seed": 42,
        "folds": fold_reports,
        "original_all_exact": all(item["original_exact"] for item in fold_reports),
        "specialist_all_exact": all(item["specialist_exact"] for item in fold_reports),
        "decision": (
            "PASS"
            if all(item["original_exact"] and item["specialist_exact"] for item in fold_reports)
            else "FAIL"
        ),
        "scope_note": (
            "This proves that identity projection itself preserves both selectors. "
            "It does not authorize old selector scores for semantic-family-v3."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit selector parity without training.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    args = parser.parse_args()
    report = run_old_protocol_parity(args.data.resolve(), args.oof.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
