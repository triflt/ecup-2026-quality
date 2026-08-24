from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "677"
OUTER_SCREEN_FOLD = 0
INNER_FOLDS = (1, 2, 3, 4)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_source_runtime(path: Path, expected_fold: int) -> dict[str, Any]:
    audit_path = path / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(audit)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"source runtime self-hash mismatch for fold {expected_fold}")
    expected = {
        "outer_fold": expected_fold,
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f"source runtime contract mismatch for fold {expected_fold}")
    for name in ("train.jsonl", "validation.jsonl"):
        if audit["output_sha256"][name] != sha256_file(path / name):
            raise ValueError(f"source runtime checksum mismatch: fold={expected_fold}, file={name}")
    return audit


def trim_to_frozen_micro_batch(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    remainder = len(rows) % 4
    if remainder == 0:
        return rows, []
    removable = [
        index
        for index in range(len(rows) - 1, -1, -1)
        if rows[index]["category"] == "БАД" and int(rows[index]["label"]) == 0
    ]
    if len(removable) < remainder:
        raise ValueError("not enough BAD negatives for deterministic tail trim")
    removed_indices = set(removable[:remainder])
    removed_ids = [str(rows[index]["id"]) for index in sorted(removed_indices)]
    return [row for index, row in enumerate(rows) if index not in removed_indices], removed_ids


def build(
    *, outer_runtime: Path, validation_runtimes: list[Path], output_dir: Path
) -> dict[str, Any]:
    if len(validation_runtimes) != len(INNER_FOLDS):
        raise ValueError("four validation runtimes are required in fold 1,2,3,4 order")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    outer_audit = validate_source_runtime(outer_runtime, OUTER_SCREEN_FOLD)
    outer_train_path = outer_runtime / "train.jsonl"
    outer_train = read_jsonl(outer_train_path)
    if any(int(row["fold"]) == OUTER_SCREEN_FOLD for row in outer_train):
        raise ValueError("outer screen fold entered source training")
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "outer_screen_fold": OUTER_SCREEN_FOLD,
        "source_train_sha256": sha256_file(outer_train_path),
        "source_runtime_contract_sha256": outer_audit["contract_sha256"],
        "inner_folds": [],
        "public_used": False,
        "sealed_rows": 0,
    }
    output_dir.mkdir(parents=True)
    for inner_fold, source_validation in zip(INNER_FOLDS, validation_runtimes, strict=True):
        validation_audit = validate_source_runtime(source_validation, inner_fold)
        validation_path = source_validation / "validation.jsonl"
        validation = read_jsonl(validation_path)
        if any(int(row["fold"]) != inner_fold for row in validation):
            raise ValueError(f"inner validation fold mismatch: {inner_fold}")
        if any("label" in row or "evidence_target" in row for row in validation):
            raise ValueError("inner validation supervision is forbidden")
        untrimmed = [row for row in outer_train if int(row["fold"]) != inner_fold]
        train, removed_ids = trim_to_frozen_micro_batch(untrimmed)
        if any(int(row["fold"]) in {OUTER_SCREEN_FOLD, inner_fold} for row in train):
            raise ValueError("outer or inner validation entered nested training")
        local_dir = output_dir / f"inner_fold{inner_fold}"
        local_dir.mkdir()
        train_path = local_dir / "train.jsonl"
        local_validation_path = local_dir / "validation.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(local_validation_path, validation)
        counts = Counter((row["category"], int(row["label"])) for row in train)
        audit = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "objective": "class_only_checkpoint_dynamics",
            "outer_screen_fold": OUTER_SCREEN_FOLD,
            "inner_validation_fold": inner_fold,
            "source_runtime_contract_sha256": outer_audit["contract_sha256"],
            "source_validation_contract_sha256": validation_audit["contract_sha256"],
            "source_train_sha256": sha256_file(outer_train_path),
            "source_validation_sha256": sha256_file(validation_path),
            "train_occurrences_before_tail_trim": len(untrimmed),
            "train_occurrences": len(train),
            "train_unique_ids": len({str(row["id"]) for row in train}),
            "validation_rows": len(validation),
            "tail_trimmed_ids": removed_ids,
            "train_counts": {
                "bad_negative": counts[("БАД", 0)],
                "bad_positive": counts[("БАД", 1)],
                "flammable_negative": counts[("Легковоспламеняющиеся", 0)],
                "flammable_positive": counts[("Легковоспламеняющиеся", 1)],
            },
            "train_excluded_folds": [OUTER_SCREEN_FOLD, inner_fold],
            "model_input_fields": ["category", "name", "description", "first_image"],
            "validation_labels_written": 0,
            "sealed_rows_written": 0,
            "public_used": False,
            "selected_multiset_sha256": canonical_sha256(
                sorted(Counter(str(row["id"]) for row in train).items())
            ),
            "output_sha256": {
                "train.jsonl": sha256_file(train_path),
                "validation.jsonl": sha256_file(local_validation_path),
            },
            "decision": "READY_FOR_TECHNICAL_SMOKE_AFTER_TERMINAL_659",
        }
        audit["contract_sha256"] = canonical_sha256(audit)
        audit_path = local_dir / "runtime_audit.json"
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result["inner_folds"].append(
            {
                "fold": inner_fold,
                "runtime_contract_sha256": audit["contract_sha256"],
                "runtime_audit_sha256": sha256_file(audit_path),
                "train_occurrences": len(train),
                "validation_rows": len(validation),
                "tail_trimmed_rows": len(removed_ids),
            }
        )
    result["contract_sha256"] = canonical_sha256(result)
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--outer-runtime", type=Path, required=True)
    result.add_argument("--validation-runtime", type=Path, action="append", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(
        json.dumps(
            build(
                outer_runtime=arguments.outer_runtime,
                validation_runtimes=arguments.validation_runtime,
                output_dir=arguments.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
