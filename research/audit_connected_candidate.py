from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_connected_guard_existing import GUARD, audit


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline-key", default="baseline_nested_predictions")
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=31042)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = np.load(args.predictions, allow_pickle=False)
    guard = pd.read_csv(
        GUARD,
        dtype={"id": str, "connected_component": str},
    )
    required = {
        "ids",
        "labels",
        "categories",
        "folds",
        args.baseline_key,
        args.candidate_key,
    }
    missing = required - set(arrays.files)
    if missing:
        raise ValueError(f"prediction archive is missing keys: {sorted(missing)}")
    ids = arrays["ids"].astype(str)
    labels = arrays["labels"].astype(np.int8)
    categories = arrays["categories"].astype(str)
    folds = arrays["folds"].astype(np.int8)
    if guard.id.duplicated().any() or not np.array_equal(guard.id.to_numpy(), ids):
        raise ValueError("prediction ids do not match connected guard")
    if not np.array_equal(guard.label.to_numpy(np.int8), labels):
        raise ValueError("prediction labels do not match connected guard")
    if not np.array_equal(guard.category.astype(str).to_numpy(), categories):
        raise ValueError("prediction categories do not match connected guard")
    if not np.array_equal(guard.fold.to_numpy(np.int8), folds):
        raise ValueError("prediction folds do not match connected guard")
    result = {
        "evaluation_version": "connected_family_guard_v2",
        "candidate": args.name,
        "audit": audit(
            labels=labels,
            categories=categories,
            folds=folds,
            components=guard.connected_component.astype(str).to_numpy(),
            safe=guard.safe_for_selection.astype(bool).to_numpy(),
            baseline=arrays[args.baseline_key].astype(np.int8),
            candidate=arrays[args.candidate_key].astype(np.int8),
            bootstrap=args.bootstrap,
            seed=args.seed,
        ),
        "input_sha256": {
            "predictions": sha256(args.predictions),
            "guard": sha256(GUARD),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
