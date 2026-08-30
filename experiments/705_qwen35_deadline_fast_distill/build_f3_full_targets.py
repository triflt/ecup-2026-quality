from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FLAMMABLE = "Легковоспламеняющиеся"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def consistent_score_map(rows: list[dict]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        row_id, score = str(row["id"]), float(row["score"])
        if row_id in result and result[row_id] != score:
            raise ValueError(f"inconsistent repeated teacher score for id={row_id}")
        result[row_id] = score
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--outer-target-dir", type=Path, required=True)
    parser.add_argument("--fold-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty fast target output")
    args.output_dir.mkdir(parents=True)

    train_path = args.runtime_dir / "train.jsonl"
    audit_path = args.runtime_dir / "runtime_audit.json"
    outer_path = args.outer_target_dir / "teacher_targets.jsonl"
    outer_contract_path = args.outer_target_dir / "teacher_target_contract.json"
    prediction_path = args.fold_prediction_dir / "predictions.jsonl"
    prediction_contract_path = args.fold_prediction_dir / "output_contract.json"
    runtime = json.loads(audit_path.read_text(encoding="utf-8"))
    outer_contract = json.loads(outer_contract_path.read_text(encoding="utf-8"))
    prediction_contract = json.loads(
        prediction_contract_path.read_text(encoding="utf-8")
    )
    if runtime.get("schema_version") != "exp715_full_runtime_v1":
        raise ValueError("full runtime schema mismatch")
    if runtime.get("train_sha256") != sha256(train_path):
        raise ValueError("full runtime checksum mismatch")
    if outer_contract.get("fold") != 3 or prediction_contract.get("fold") != 3:
        raise ValueError("fast targets require fold-3 teacher artifacts")
    if outer_contract.get("teacher_targets_sha256") != sha256(outer_path):
        raise ValueError("outer teacher target checksum mismatch")
    if prediction_contract.get("predictions_sha256") != sha256(prediction_path):
        raise ValueError("fold prediction checksum mismatch")

    rows = read_jsonl(train_path)
    outer = consistent_score_map(read_jsonl(outer_path))
    validation = consistent_score_map(read_jsonl(prediction_path))
    raw_scores: list[float] = []
    sources: list[str] = []
    for row in rows:
        row_id = str(row["id"])
        if str(row["category"]) != FLAMMABLE:
            raw_scores.append(0.0)
            sources.append("neutral_unconsumed_category")
        elif int(row["fold"]) == 3:
            if row_id not in validation:
                raise ValueError(f"missing fold-3 OOF score for id={row_id}")
            raw_scores.append(validation[row_id])
            sources.append("fold3_oof_validation")
        else:
            if row_id not in outer:
                raise ValueError(f"missing fold-3 in-sample score for id={row_id}")
            raw_scores.append(outer[row_id])
            sources.append("fold3_in_sample_outer_train")

    frame = pd.DataFrame(
        {
            "score": raw_scores,
            "fold": [int(row["fold"]) for row in rows],
            "category": [str(row["category"]) for row in rows],
        }
    )
    rank_scores = np.empty(len(frame), dtype=np.float64)
    for (_, _), positions in frame.groupby(["fold", "category"]).groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        rank_scores[index] = frame.iloc[index]["score"].rank(
            method="average", pct=True
        ).to_numpy(np.float64)

    target_path = args.output_dir / "teacher_targets.jsonl"
    with target_path.open("w", encoding="utf-8") as stream:
        for occurrence, (row, score, rank, source) in enumerate(
            zip(rows, raw_scores, rank_scores, sources, strict=True)
        ):
            stream.write(
                json.dumps(
                    {
                        "occurrence_index": occurrence,
                        "id": str(row["id"]),
                        "source_fold": int(row["fold"]),
                        "category": str(row["category"]),
                        "label": int(row["label"]),
                        "score": float(score),
                        "rank_score": float(rank),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    contract = {
        "schema_version": "exp705_f3_full_targets_v1",
        "experiment_id": "705",
        "teacher_experiment_id": "697",
        "teacher_source_fold": 3,
        "target_scope": "full_train_occurrences",
        "teacher_scores_are_strict_oof_by_id": False,
        "teacher_never_trained_on_target_row": False,
        "oof_source_folds": [3],
        "in_sample_source_folds": [0, 1, 2, 4],
        "rank_score_calibration": "category_and_source_fold_percentile_average_ties",
        "rows": len(rows),
        "unique_ids": len({str(row["id"]) for row in rows}),
        "runtime_audit_sha256": sha256(audit_path),
        "train_runtime_sha256": sha256(train_path),
        "teacher_targets_sha256": sha256(target_path),
        "teacher_adapter_model_sha256": outer_contract["adapter_model_sha256"],
        "outer_target_contract_sha256": sha256(outer_contract_path),
        "fold_prediction_contract_sha256": sha256(prediction_contract_path),
        "source_counts": {
            str(key): int(value)
            for key, value in pd.Series(sources).value_counts().items()
        },
        "decision": "DEADLINE_FAST_F3_TARGETS_FROZEN",
    }
    (args.output_dir / "teacher_target_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
