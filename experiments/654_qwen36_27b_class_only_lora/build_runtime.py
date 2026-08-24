from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "654"
SOURCE_EXPERIMENT_ID = "641"
SOURCE_GRID_SHA256 = "aad63f99ee9ddd5ecfa133575ea5cf1a803c11ab3fd26b5328d894f8631b3224"
EXPECTED_TRAIN_OCCURRENCES = 4892
EXPECTED_VALIDATION_ROWS = 2224
SCREEN_FOLDS = {0, 3}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def build(source: Path, output: Path, *, fold: int) -> dict[str, Any]:
    if fold not in SCREEN_FOLDS:
        raise ValueError("only frozen screen folds 0 and 3 are open")
    if output.exists():
        raise FileExistsError("refusing to overwrite a runtime directory")
    train_path = source / "train.jsonl"
    validation_path = source / "validation.jsonl"
    audit_path = source / "runtime_audit.json"
    for path in (train_path, validation_path, audit_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    source_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source_payload = dict(source_audit)
    source_digest = source_payload.pop("contract_sha256", None)
    if source_digest != canonical_sha256(source_payload):
        raise ValueError("source runtime audit self-hash mismatch")
    expected = {
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "grid_contract_sha256": SOURCE_GRID_SHA256,
        "objective": "class_only",
        "outer_fold": fold,
        "train_occurrences": EXPECTED_TRAIN_OCCURRENCES,
        "validation_rows": EXPECTED_VALIDATION_ROWS,
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "outer_validation_occurrences": 0,
        "decision": "GO",
    }
    mismatch = {
        key: {"expected": value, "actual": source_audit.get(key)}
        for key, value in expected.items()
        if source_audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"source runtime contract mismatch: {mismatch}")
    if source_audit.get("output_sha256") != {
        "train.jsonl": sha256_file(train_path),
        "validation.jsonl": sha256_file(validation_path),
    }:
        raise ValueError("source runtime file checksum mismatch")
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    if len(train) != EXPECTED_TRAIN_OCCURRENCES or len(validation) != EXPECTED_VALIDATION_ROWS:
        raise ValueError("runtime row count mismatch")
    if any("label" not in row or int(row["fold"]) == fold for row in train):
        raise ValueError("invalid training supervision or outer-fold leakage")
    if any("label" in row or "evidence_target" in row for row in validation):
        raise ValueError("validation supervision is forbidden")
    if any(int(row["fold"]) != fold for row in validation):
        raise ValueError("validation fold mismatch")
    output.mkdir(parents=True)
    output_train = output / "train.jsonl"
    output_validation = output / "validation.jsonl"
    shutil.copyfile(train_path, output_train)
    shutil.copyfile(validation_path, output_validation)
    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_contract_sha256": source_digest,
        "data_version": "competition_train_v1",
        "evaluation_version": "semantic_family_v3",
        "objective": "class_only",
        "outer_fold": fold,
        "train_occurrences": len(train),
        "train_unique_ids": len({str(row["id"]) for row in train}),
        "validation_rows": len(validation),
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "public_used": False,
        "selected_multiset_sha256": source_audit["selected_multiset_sha256"],
        "model_input_view_sha256": source_audit["model_input_view_sha256"],
        "output_sha256": {
            "train.jsonl": sha256_file(output_train),
            "validation.jsonl": sha256_file(output_validation),
        },
        "decision": "GO_AFTER_653",
    }
    audit["contract_sha256"] = canonical_sha256(audit)
    (output / "runtime_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-runtime", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(build(args.source_runtime, args.output_dir, fold=args.fold), indent=2))
