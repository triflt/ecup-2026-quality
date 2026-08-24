from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from contract import (
    CONTROL_EXPERIMENT_ID,
    EXPECTED_RUNTIME_CONTRACT,
    EXPECTED_TRAIN_OCCURRENCES,
    EXPECTED_TRAIN_SHA256,
    EXPECTED_VALIDATION_ROWS,
    EXPECTED_VALIDATION_SHA256,
    EXPERIMENT_ID,
    GRID_CONTRACT_SHA256,
    SCREEN_FOLDS,
    canonical_sha256,
    sha256_file,
    verify_self_hash,
)


def build(source: Path, output: Path, fold: int) -> dict:
    if fold not in SCREEN_FOLDS:
        raise ValueError("only frozen screen folds 0/3 are allowed")
    if output.exists():
        raise FileExistsError("refusing to overwrite runtime")
    audit_path = source / "runtime_audit.json"
    train_path = source / "train.jsonl"
    validation_path = source / "validation.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    verify_self_hash(audit)
    expected = {
        "experiment_id": CONTROL_EXPERIMENT_ID,
        "objective": "class_only",
        "outer_fold": fold,
        "grid_contract_sha256": GRID_CONTRACT_SHA256,
        "contract_sha256": EXPECTED_RUNTIME_CONTRACT[fold],
        "train_occurrences": EXPECTED_TRAIN_OCCURRENCES,
        "validation_rows": EXPECTED_VALIDATION_ROWS[fold],
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "decision": "GO",
    }
    mismatch = {
        key: {"expected": value, "actual": audit.get(key)}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"source runtime mismatch: {mismatch}")
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA256[fold]:
        raise ValueError("source train checksum mismatch")
    if sha256_file(validation_path) != EXPECTED_VALIDATION_SHA256[fold]:
        raise ValueError("source validation checksum mismatch")
    validation = [json.loads(line) for line in validation_path.read_text().splitlines()]
    if any(set(row) & {"label", "target", "gold", "answer"} for row in validation):
        raise ValueError("validation supervision is forbidden")
    output.mkdir(parents=True)
    shutil.copy2(train_path, output / "train.jsonl")
    shutil.copy2(validation_path, output / "validation.jsonl")
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "control_experiment_id": CONTROL_EXPERIMENT_ID,
        "outer_fold": fold,
        "objective": "class_only",
        "source_runtime_contract_sha256": audit["contract_sha256"],
        "source_grid_contract_sha256": GRID_CONTRACT_SHA256,
        "source_model_input_view_sha256": audit["model_input_view_sha256"],
        "train_occurrences": EXPECTED_TRAIN_OCCURRENCES,
        "validation_rows": EXPECTED_VALIDATION_ROWS[fold],
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "output_sha256": {
            "train.jsonl": sha256_file(output / "train.jsonl"),
            "validation.jsonl": sha256_file(output / "validation.jsonl"),
        },
        "changed_factor": "qwen35_4b_to_gemma4_e4b",
        "decision": "GO_TECHNICAL_SMOKE_ONLY",
    }
    result["contract_sha256"] = canonical_sha256(result)
    (output / "runtime_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=SCREEN_FOLDS, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_runtime, args.output_dir, args.fold), indent=2))
