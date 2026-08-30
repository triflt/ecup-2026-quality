from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
EXPECTED_FOLDS_SHA256 = "03baaa25bd5a3aef6ad94e02067cccda114041f98d7a35a9e06330a425166e4d"
EXPECTED_TEACHER_ROWS = 12971
EXPECTED_SCORE_CALIBRATION = (
    "label_blind_percentile_rank_within_fold_and_category_average_ties"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--teacher-oof", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty target output")
    runtime_path = args.runtime_dir / "train.jsonl"
    audit_path = args.runtime_dir / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "exp715_full_runtime_v1":
        raise ValueError("full runtime schema mismatch")
    if audit.get("train_sha256") != sha256(runtime_path):
        raise ValueError("full runtime checksum mismatch")
    teacher_report = json.loads(args.teacher_report.read_text(encoding="utf-8"))
    expected_report = {
        "schema_version": "exp697_teacher_oof_v2",
        "experiment_id": "697",
        "evaluation": "nested_grouped_fold_category_percentile_rank_v2",
        "score_calibration": EXPECTED_SCORE_CALIBRATION,
        "raw_teacher_score_preserved": True,
        "rows": EXPECTED_TEACHER_ROWS,
        "unique_ids": EXPECTED_TEACHER_ROWS,
        "data_sha256": EXPECTED_DATA_SHA256,
        "folds_sha256": EXPECTED_FOLDS_SHA256,
    }
    mismatch = {
        key: {"expected": value, "actual": teacher_report.get(key)}
        for key, value in expected_report.items()
        if teacher_report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"teacher OOF report mismatch: {mismatch}")
    if teacher_report.get("teacher_oof_sha256") != sha256(args.teacher_oof):
        raise ValueError("teacher OOF/report checksum mismatch")

    train = read_jsonl(runtime_path)
    teacher = pd.read_csv(args.teacher_oof, dtype={"id": str})
    required_columns = {
        "id",
        "fold",
        "category",
        "label",
        "teacher_score",
        "teacher_rank",
        "teacher_prediction",
    }
    if set(teacher.columns) != required_columns:
        raise ValueError("teacher OOF column schema mismatch")
    if (
        len(teacher) != EXPECTED_TEACHER_ROWS
        or teacher["id"].nunique() != EXPECTED_TEACHER_ROWS
        or teacher["id"].duplicated().any()
    ):
        raise ValueError("teacher OOF does not cover canonical unique IDs")
    if set(teacher["fold"].astype(int)) != set(range(5)):
        raise ValueError("teacher OOF fold coverage mismatch")
    if not set(teacher["label"].astype(int)).issubset({0, 1}) or not set(
        teacher["teacher_prediction"].astype(int)
    ).issubset({0, 1}):
        raise ValueError("teacher OOF binary columns are invalid")
    raw_values = teacher["teacher_score"].to_numpy(np.float64)
    rank_values = teacher["teacher_rank"].to_numpy(np.float64)
    if (
        not np.isfinite(raw_values).all()
        or not np.isfinite(rank_values).all()
        or np.any(rank_values <= 0.0)
        or np.any(rank_values > 1.0)
    ):
        raise ValueError("teacher OOF scores are invalid")
    expected_rank = teacher.groupby(["fold", "category"], sort=False)[
        "teacher_score"
    ].rank(method="average", pct=True)
    if not np.allclose(
        rank_values,
        expected_rank.to_numpy(np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("teacher OOF rank calibration mismatch")
    score_by_id = dict(zip(teacher["id"], teacher["teacher_score"], strict=True))
    rank_by_id = dict(zip(teacher["id"], teacher["teacher_rank"], strict=True))
    fold_by_id = dict(zip(teacher["id"], teacher["fold"].astype(int), strict=True))
    args.output_dir.mkdir(parents=True)
    target_path = args.output_dir / "teacher_targets.jsonl"
    with target_path.open("w", encoding="utf-8") as stream:
        for occurrence_index, row in enumerate(train):
            row_id = str(row["id"])
            score = float(score_by_id[row_id])
            if not math.isfinite(score) or fold_by_id[row_id] != int(row["fold"]):
                raise ValueError("teacher OOF occurrence binding mismatch")
            target = {
                "occurrence_index": occurrence_index,
                "id": row_id,
                "source_fold": int(row["fold"]),
                "category": str(row["category"]),
                "label": int(row["label"]),
                "score": score,
                "rank_score": float(rank_by_id[row_id]),
            }
            stream.write(json.dumps(target, ensure_ascii=False, sort_keys=True) + "\n")
    contract = {
        "schema_version": "exp715_full_oof_targets_v1",
        "experiment_id": "715",
        "teacher_experiment_id": "697",
        "target_scope": "full_train_occurrences",
        "teacher_scores_are_strict_oof_by_id": True,
        "teacher_never_trained_on_target_row": True,
        "rank_score_calibration": "category_and_source_fold_percentile_average_ties",
        "rows": len(train),
        "unique_ids": len({row["id"] for row in train}),
        "runtime_audit_sha256": sha256(audit_path),
        "train_runtime_sha256": sha256(runtime_path),
        "teacher_oof_sha256": sha256(args.teacher_oof),
        "teacher_oof_report_sha256": sha256(args.teacher_report),
        "teacher_targets_sha256": sha256(target_path),
        "decision": "FULL_OOF_TARGETS_FROZEN",
    }
    (args.output_dir / "teacher_target_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
