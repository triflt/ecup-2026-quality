from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "681"
CONTROL_EXPERIMENT_ID = "641"
TEACHER_EXPERIMENT_ID = "662"
FLAMMABLE = "Легковоспламеняющиеся"
KEY_FIELDS = ("global_index", "id", "category", "fold", "occurrence_index")
SCORE_FIELDS = {*KEY_FIELDS, "score"}
FORBIDDEN = {"label", "target", "gold", "answer", "sealed", "public"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode())


def read_jsonl_bytes(payload: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.decode().splitlines()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_bytes(path.read_bytes())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def verify_self_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    digest = body.pop(field, None)
    if digest != canonical_sha256(body):
        raise ValueError(f"{field} self-hash mismatch")
    return str(digest)


def load_teacher(
    archive_path: Path, teacher_runtime: Path, fold: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime_path = teacher_runtime / "runtime_audit.json"
    input_path = teacher_runtime / "score_input.jsonl"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_digest = verify_self_hash(runtime, "contract_sha256")
    if (
        runtime.get("experiment_id") != TEACHER_EXPERIMENT_ID
        or int(runtime.get("outer_fold", -1)) != fold
        or runtime.get("mode") != "full"
        or runtime.get("score_semantics") != "raw_last_token_logit_1_minus_logit_0"
        or int(runtime.get("train_outer_validation_overlap", -1)) != 0
        or int(runtime.get("source_validation_rows_bundled", -1)) != 0
        or int(runtime.get("source_validation_labels_bundled", -1)) != 0
        or int(runtime.get("labels_bundled", -1)) != 0
        or int(runtime.get("sealed_rows", -1)) != 0
        or runtime.get("public_used") is not False
        or runtime.get("decision") != "GO_TEACHER_TRAIN_TARGET_SCORING"
    ):
        raise ValueError("teacher runtime violates the outer-safe full-score contract")
    input_payload = input_path.read_bytes()
    if sha256_bytes(input_payload) != runtime.get("score_input_sha256"):
        raise ValueError("teacher score-input SHA mismatch")
    input_rows = read_jsonl_bytes(input_payload)
    if any(FORBIDDEN.intersection(row) for row in input_rows):
        raise ValueError("teacher score input contains forbidden supervision")

    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("corrupt teacher archive")
        if set(archive.namelist()) != {"report.json", "teacher_scores.jsonl"}:
            raise ValueError("teacher archive schema mismatch")
        report = json.loads(archive.read("report.json"))
        score_payload = archive.read("teacher_scores.jsonl")
    report_digest = verify_self_hash(report, "report_contract_sha256")
    expected_report = {
        "experiment_id": TEACHER_EXPERIMENT_ID,
        "source_experiment_id": "654",
        "outer_fold": fold,
        "mode": "full",
        "rows": runtime["rows"],
        "score_semantics": "raw_last_token_logit_1_minus_logit_0",
        "runtime_contract_sha256": runtime_digest,
        "teacher_adapter_manifest_sha256": runtime["teacher_adapter_manifest_sha256"],
        "ordered_occurrence_key_sha256": runtime["ordered_occurrence_key_sha256"],
        "validation_rows_read": 0,
        "validation_labels_read": 0,
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "decision": "READY_FOR_FAIL_CLOSED_ACCEPTANCE",
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise ValueError("teacher report contract mismatch")
    scores = read_jsonl_bytes(score_payload)
    if len(scores) != len(input_rows) or len(scores) != int(runtime["rows"]):
        raise ValueError("teacher score row count mismatch")
    if any(set(row) != SCORE_FIELDS for row in scores):
        raise ValueError("teacher score schema mismatch")
    input_keys = [[row[key] for key in KEY_FIELDS] for row in input_rows]
    score_keys = [[row[key] for key in KEY_FIELDS] for row in scores]
    if score_keys != input_keys or canonical_sha256(score_keys) != runtime["ordered_occurrence_key_sha256"]:
        raise ValueError("teacher occurrence binding mismatch")
    if any(not math.isfinite(float(row["score"])) for row in scores):
        raise ValueError("non-finite teacher score")
    score_sha = sha256_bytes(score_payload)
    if report.get("teacher_scores_sha256") != score_sha:
        raise ValueError("teacher score payload SHA mismatch")
    return scores, {
        "archive_sha256": sha256_file(archive_path),
        "report_contract_sha256": report_digest,
        "runtime_contract_sha256": runtime_digest,
        "teacher_scores_sha256": score_sha,
        "ordered_occurrence_key_sha256": runtime["ordered_occurrence_key_sha256"],
        "teacher_adapter_manifest_sha256": runtime["teacher_adapter_manifest_sha256"],
    }


def build(
    base_runtime: Path,
    teacher_runtime: Path,
    teacher_artifact: Path,
    output: Path,
    fold: int,
) -> dict[str, Any]:
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    if output.exists():
        raise FileExistsError("refusing to replace an existing runtime")
    base_audit = json.loads((base_runtime / "runtime_audit.json").read_text(encoding="utf-8"))
    verify_self_hash(base_audit, "contract_sha256")
    train_path = base_runtime / "train.jsonl"
    validation_path = base_runtime / "validation.jsonl"
    if base_audit.get("output_sha256") != {
        "train.jsonl": sha256_file(train_path),
        "validation.jsonl": sha256_file(validation_path),
    }:
        raise ValueError("base runtime payload SHA mismatch")
    if (
        base_audit.get("experiment_id") != CONTROL_EXPERIMENT_ID
        or base_audit.get("objective") != "class_only"
        or int(base_audit.get("outer_fold", -1)) != fold
        or base_audit.get("decision") != "GO"
    ):
        raise ValueError("unexpected base runtime contract")
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    if any(row.get("category") != FLAMMABLE or int(row.get("fold", fold)) == fold for row in train_rows):
        raise ValueError("base train rows are not flammable-only outer training")
    if any(FORBIDDEN.intersection(row) for row in validation_rows):
        raise ValueError("outer validation contains supervision")

    all_scores, teacher_manifest = load_teacher(teacher_artifact, teacher_runtime, fold)
    scores = [row for row in all_scores if row["category"] == FLAMMABLE]
    train_keys = [[row[key] for key in KEY_FIELDS] for row in train_rows]
    score_keys = [[row[key] for key in KEY_FIELDS] for row in scores]
    if train_keys != score_keys:
        raise ValueError("teacher scores do not bind exactly to student train occurrences")
    joined = []
    for row, score in zip(train_rows, scores, strict=True):
        item = dict(row)
        item["teacher_score"] = float(score["score"])
        joined.append(item)

    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", joined)
    shutil.copyfile(validation_path, output / "validation.jsonl")
    audit = dict(base_audit)
    audit.pop("contract_sha256", None)
    audit.update(
        {
            "distillation_experiment_id": EXPERIMENT_ID,
            "changed_factor": "hard_bce_to_fixed_hard_plus_teacher_soft_bce",
            "teacher_experiment_id": TEACHER_EXPERIMENT_ID,
            "teacher_outer_fold": fold,
            "teacher_target_scope": "outer_train_in_sample_outer_validation_unread",
            "teacher_manifest": teacher_manifest,
            "temperature": 2.0,
            "soft_loss_weight": 0.5,
            "ordinary_oof_merge_used": False,
            "outer_validation_teacher_overlap": 0,
            "output_sha256": {
                "train.jsonl": sha256_file(output / "train.jsonl"),
                "validation.jsonl": sha256_file(output / "validation.jsonl"),
            },
        }
    )
    audit["contract_sha256"] = canonical_sha256(audit)
    (output / "runtime_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-runtime", type=Path, required=True)
    result.add_argument("--teacher-runtime", type=Path, required=True)
    result.add_argument("--teacher-artifact", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            build(
                args.base_runtime,
                args.teacher_runtime,
                args.teacher_artifact,
                args.output,
                args.fold,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
