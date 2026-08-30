from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path


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


def average_tie_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_position = (start + end - 1) / 2.0
        rank = average_position / max(1, len(values) - 1)
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def atomic_write_jsonl(rows: list[dict], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-experiment", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--folds-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "validation/grouped_text_v1/folds.csv",
    )
    args = parser.parse_args()
    folds = sorted(set(args.folds))
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("folds must be a non-empty subset of 0..4")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty OOF runtime")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.folds_csv.open(encoding="utf-8", newline="") as stream:
        fold_contract = {str(row["id"]): row for row in csv.DictReader(stream)}

    rows: list[dict] = []
    scores: list[float] = []
    sources: list[dict] = []
    seen_ids: set[str] = set()
    for fold in folds:
        runtime_path = args.teacher_experiment / ".local" / f"runtime-f{fold}" / "validation.jsonl"
        prediction_path = args.teacher_experiment / ".local" / f"output-f{fold}" / "predictions.jsonl"
        contract_path = args.teacher_experiment / ".local" / f"output-f{fold}" / "output_contract.json"
        if not contract_path.is_file():
            raise FileNotFoundError(f"teacher fold {fold} contract is missing")
        local_rows = read_jsonl(runtime_path)
        local_predictions = read_jsonl(prediction_path)
        if len(local_rows) != len(local_predictions):
            raise ValueError(f"fold {fold} row/prediction count mismatch")
        for row, prediction in zip(local_rows, local_predictions, strict=True):
            row_id = str(row["id"])
            expected = fold_contract.get(row_id)
            if expected is None:
                raise ValueError(f"id absent from fold contract: {row_id}")
            if int(row["fold"]) != fold or int(prediction["fold"]) != fold:
                raise ValueError(f"fold {fold} binding mismatch")
            if int(expected["fold"]) != fold or str(expected["category"]) != str(row["category"]):
                raise ValueError(f"fold contract mismatch for id={row_id}")
            if str(prediction["id"]) != row_id:
                raise ValueError(f"fold {fold} id/order mismatch")
            if row_id in seen_ids:
                raise ValueError(f"duplicate OOF id: {row_id}")
            value = float(prediction["score"])
            if not math.isfinite(value):
                raise ValueError("non-finite teacher score")
            seen_ids.add(row_id)
            bound_row = dict(row)
            bound_row["label"] = int(expected["label"])
            rows.append(bound_row)
            scores.append(value)
        sources.append(
            {
                "fold": fold,
                "runtime_sha256": sha256(runtime_path),
                "predictions_sha256": sha256(prediction_path),
                "output_contract_sha256": sha256(contract_path),
                "rows": len(local_rows),
            }
        )

    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(int(row["fold"]), str(row["category"]))].append(index)
    rank_scores = [0.0] * len(rows)
    for indices in grouped.values():
        local_ranks = average_tie_ranks([scores[index] for index in indices])
        for index, rank in zip(indices, local_ranks, strict=True):
            rank_scores[index] = rank

    targets = [
        {
            "occurrence_index": occurrence,
            "id": str(row["id"]),
            "source_fold": int(row["fold"]),
            "category": str(row["category"]),
            "label": int(row["label"]),
            "score": score,
            "rank_score": rank,
        }
        for occurrence, (row, score, rank) in enumerate(
            zip(rows, scores, rank_scores, strict=True)
        )
    ]
    runtime_dir = args.output_dir / "runtime"
    target_dir = args.output_dir / "targets"
    runtime_dir.mkdir()
    target_dir.mkdir()
    runtime_path = runtime_dir / "train.jsonl"
    target_path = target_dir / "teacher_targets.jsonl"
    atomic_write_jsonl(rows, runtime_path)
    atomic_write_jsonl(targets, target_path)
    contract = {
        "schema_version": "exp706_oof_teacher_targets_v1",
        "experiment_id": "706",
        "source_experiment_id": "697",
        "folds": folds,
        "strict_oof": True,
        "teacher_trained_without_each_target_row": True,
        "rows": len(rows),
        "unique_ids": len(seen_ids),
        "flammable_rows": sum(str(row["category"]) == FLAMMABLE for row in rows),
        "runtime_sha256": sha256(runtime_path),
        "teacher_targets_sha256": sha256(target_path),
        "sources": sources,
    }
    contract_path = target_dir / "teacher_target_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
