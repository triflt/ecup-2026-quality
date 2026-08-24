from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


EXPECTED_TRAIN_OCCURRENCES = {0: 4892, 1: 4894, 2: 4892, 3: 4892, 4: 4894}
EXPECTED_VALIDATION_ROWS = {0: 2224, 1: 2223, 2: 2224, 3: 2224, 4: 2223}


def verify(path: Path, *, fold: int, expected_rows: int | None = None) -> dict[str, Any]:
    if fold not in EXPECTED_VALIDATION_ROWS:
        raise ValueError("fold must be 0..4")
    if expected_rows is None:
        expected_rows = EXPECTED_VALIDATION_ROWS[fold]
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
        names = archive.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("unsafe ZIP member")
        if "report.json" not in names or "predictions.jsonl" not in names:
            raise ValueError("required artifact member missing")
        report = json.loads(archive.read("report.json"))
        predictions_payload = archive.read("predictions.jsonl")
        predictions = [json.loads(line) for line in predictions_payload.decode().splitlines()]
        adapter_files = {
            name.removeprefix("adapter/"): sha256_bytes(archive.read(name))
            for name in names
            if name.startswith("adapter/") and not name.endswith("/")
        }
    expected = {
        "experiment_id": "654",
        "model_revision": MODEL_REVISION,
        "outer_fold": fold,
        "train_occurrences": EXPECTED_TRAIN_OCCURRENCES[fold],
        "validation_rows": expected_rows,
        "optimizer_steps": 306,
        "threshold": 0.0,
        "threshold_tuned": False,
        "validation_labels_written": 0,
        "sealed_rows": 0,
        "public_used": False,
        "cpu_or_disk_offload": False,
        "decision": "READY_FOR_FROZEN_EVALUATION",
    }
    mismatch = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"report contract mismatch: {mismatch}")
    if len(predictions) != expected_rows:
        raise ValueError("prediction row count mismatch")
    required = {"global_index", "id", "fold", "category", "score", "prediction"}
    if any(set(row) != required for row in predictions):
        raise ValueError("prediction schema mismatch")
    if len({int(row["global_index"]) for row in predictions}) != expected_rows:
        raise ValueError("duplicate prediction index")
    if any(int(row["fold"]) != fold for row in predictions):
        raise ValueError("prediction fold mismatch")
    if any(not math.isfinite(float(row["score"])) for row in predictions):
        raise ValueError("non-finite prediction score")
    if any(int(row["prediction"]) != int(float(row["score"]) >= 0.0) for row in predictions):
        raise ValueError("prediction differs from zero threshold")
    if report.get("predictions_sha256") != sha256_bytes(predictions_payload):
        raise ValueError("prediction checksum mismatch")
    if report.get("adapter_manifest") != adapter_files:
        raise ValueError("adapter manifest mismatch")
    required_adapter = {"adapter_config.json", "adapter_model.safetensors"}
    if not required_adapter <= set(adapter_files):
        raise ValueError("adapter files missing")
    return {
        "schema_version": 1,
        "experiment_id": "654",
        "outer_fold": fold,
        "rows": len(predictions),
        "finite_scores": len(predictions),
        "archive_sha256": sha256_bytes(path.read_bytes()),
        "predictions_sha256": sha256_bytes(predictions_payload),
        "adapter_files": len(adapter_files),
        "decision": "PASS",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--expected-rows", type=int)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(verify(args.archive, fold=args.fold, expected_rows=args.expected_rows), indent=2)
    )
