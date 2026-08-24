from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "662"
SOURCE_EXPERIMENT_ID = "654"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
EXPECTED_OCCURRENCES = {0: 4892, 1: 4894, 2: 4892, 3: 4892, 4: 4894}
INPUT_FIELDS = (
    "global_index",
    "id",
    "category",
    "fold",
    "occurrence_index",
    "name",
    "description",
    "image_url",
)
KEY_FIELDS = ("global_index", "id", "category", "fold", "occurrence_index")
FORBIDDEN_FIELDS = {
    "label",
    "target",
    "gold",
    "answer",
    "evidence_target",
    "sealed",
    "public",
    "is_banned",
    "y_true",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def adapter_manifest(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def build(
    *,
    source_runtime: Path,
    accepted_artifact: Path,
    output: Path,
    fold: int,
    limit: int | None,
) -> dict[str, Any]:
    if fold not in EXPECTED_OCCURRENCES:
        raise ValueError("fold must be 0..4")
    if output.exists():
        raise FileExistsError("refusing to overwrite a runtime directory")
    train_path = source_runtime / "train.jsonl"
    validation_path = source_runtime / "validation.jsonl"
    source_audit_path = source_runtime / "runtime_audit.json"
    report_path = accepted_artifact / "report.json"
    acceptance_path = accepted_artifact / "acceptance_audit.json"
    adapter_path = accepted_artifact / "adapter"
    for path in (
        train_path,
        validation_path,
        source_audit_path,
        report_path,
        acceptance_path,
        adapter_path / "adapter_config.json",
        adapter_path / "adapter_model.safetensors",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    source_payload = dict(source_audit)
    source_contract = source_payload.pop("contract_sha256", None)
    if source_contract != canonical_sha256(source_payload):
        raise ValueError("source runtime self-hash mismatch")
    source_train_sha = sha256_file(train_path)
    source_validation_sha = sha256_file(validation_path)
    expected_source = {
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "outer_fold": fold,
        "objective": "class_only",
        "train_occurrences": EXPECTED_OCCURRENCES[fold],
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "public_used": False,
    }
    if any(source_audit.get(key) != value for key, value in expected_source.items()):
        raise ValueError("source runtime contract mismatch")
    if source_audit.get("output_sha256") != {
        "train.jsonl": source_train_sha,
        "validation.jsonl": source_validation_sha,
    }:
        raise ValueError("source runtime file SHA mismatch")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    manifest = adapter_manifest(adapter_path)
    expected_report = {
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "model_revision": MODEL_REVISION,
        "outer_fold": fold,
        "runtime_contract_sha256": source_contract,
        "decision": "READY_FOR_FROZEN_EVALUATION",
        "cpu_or_disk_offload": False,
        "validation_labels_written": 0,
        "sealed_rows": 0,
        "public_used": False,
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise ValueError("accepted teacher report mismatch")
    if report.get("adapter_manifest") != manifest:
        raise ValueError("accepted teacher adapter manifest mismatch")
    if not (
        acceptance.get("experiment_id") == SOURCE_EXPERIMENT_ID
        and acceptance.get("outer_fold") == fold
        and acceptance.get("decision") == "PASS"
        and acceptance.get("exact_runtime_binding") is True
    ):
        raise ValueError("teacher acceptance audit mismatch")

    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    if len(train_rows) != EXPECTED_OCCURRENCES[fold]:
        raise ValueError("source train occurrence count mismatch")
    if any("label" not in row or int(row["fold"]) == fold for row in train_rows):
        raise ValueError("source train supervision or outer-fold leakage mismatch")
    if any(FORBIDDEN_FIELDS.intersection(row) for row in validation_rows):
        raise ValueError("outer validation contains forbidden supervision")
    train_indices = {int(row["global_index"]) for row in train_rows}
    validation_indices = {int(row["global_index"]) for row in validation_rows}
    if train_indices & validation_indices:
        raise ValueError("source train overlaps outer validation")
    if [int(row["occurrence_index"]) for row in train_rows] != list(range(len(train_rows))):
        raise ValueError("source occurrence order is not contiguous")

    selected = train_rows if limit is None else train_rows[:limit]
    if not selected or (limit is not None and len(selected) != limit):
        raise ValueError("invalid scoring limit")
    score_rows = [{field: row[field] for field in INPUT_FIELDS} for row in selected]
    if any(FORBIDDEN_FIELDS.intersection(row) for row in score_rows):
        raise ValueError("scoring input contains forbidden supervision")
    keys = [[row[field] for field in KEY_FIELDS] for row in score_rows]
    if len({tuple(key) for key in keys}) != len(keys):
        raise ValueError("duplicate occurrence key")

    output.mkdir(parents=True)
    score_input = output / "score_input.jsonl"
    with score_input.open("w", encoding="utf-8") as stream:
        for row in score_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "data_version": "competition_train_v1",
        "outer_fold": fold,
        "mode": "smoke8" if limit == 8 else "full",
        "rows": len(score_rows),
        "source_train_occurrences": len(train_rows),
        "source_train_sha256": source_train_sha,
        "source_runtime_contract_sha256": source_contract,
        "source_validation_rows_bundled": 0,
        "source_validation_labels_bundled": 0,
        "train_outer_validation_overlap": 0,
        "labels_bundled": 0,
        "sealed_rows": 0,
        "public_used": False,
        "input_fields": list(INPUT_FIELDS),
        "key_fields": list(KEY_FIELDS),
        "ordered_occurrence_key_sha256": canonical_sha256(keys),
        "score_input_sha256": sha256_file(score_input),
        "teacher_report_sha256": sha256_file(report_path),
        "teacher_acceptance_audit_sha256": sha256_file(acceptance_path),
        "teacher_adapter_manifest": manifest,
        "teacher_adapter_manifest_sha256": canonical_sha256(manifest),
        "model_revision": MODEL_REVISION,
        "score_semantics": "raw_last_token_logit_1_minus_logit_0",
        "decision": "GO_TEACHER_TRAIN_TARGET_SCORING",
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
    result.add_argument("--accepted-artifact", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--limit", type=int)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            build(
                source_runtime=args.source_runtime,
                accepted_artifact=args.accepted_artifact,
                output=args.output_dir,
                fold=args.fold,
                limit=args.limit,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
