from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


CATEGORY = "Легковоспламеняющиеся"
SALT = "qwen35-flammable-recall-v420"
POSITIVE_REPEATS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*values: object) -> str:
    payload = "|".join(str(value) for value in (SALT, *values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_stage(
    data_by_id: dict[str, dict[str, str]],
    guard_rows: list[dict[str, str]],
    stage: str,
    holdout_fold: int | None,
) -> dict[str, object]:
    eligible = []
    for row in guard_rows:
        if row["category"] != CATEGORY or row["safe_for_selection"].lower() != "true":
            continue
        if holdout_fold is not None and int(row["fold"]) == holdout_fold:
            continue
        eligible.append(row)

    positive_ids = sorted(row["id"] for row in eligible if int(row["label"]) == 1)
    negative_pool = [row["id"] for row in eligible if int(row["label"]) == 0]
    negative_count = len(positive_ids) * POSITIVE_REPEATS
    if len(negative_pool) < negative_count:
        raise ValueError(
            f"{stage}: need {negative_count} unique negatives, found {len(negative_pool)}"
        )
    negative_ids = sorted(
        negative_pool, key=lambda item_id: stable_key("negative", stage, item_id)
    )[:negative_count]

    records = []
    for item_id in positive_ids:
        for repeat in range(POSITIVE_REPEATS):
            records.append((item_id, 1, repeat))
    records.extend((item_id, 0, 0) for item_id in negative_ids)
    records.sort(key=lambda item: stable_key("record", stage, item[0], item[1], item[2]))
    ordered_record_ids = [item[0] for item in records]

    for item_id, expected_label, _ in records:
        row = data_by_id[item_id]
        if row["category"] != CATEGORY or int(row["label"]) != expected_label:
            raise ValueError(f"{stage}: data/manifest mismatch for id={item_id}")

    record_digest = hashlib.sha256(
        "\n".join(ordered_record_ids).encode("utf-8")
    ).hexdigest()
    return {
        "stage": stage,
        "holdout_fold": holdout_fold,
        "selection_scope": "connected_family_guard_v2 safe training rows only",
        "positive_ids": positive_ids,
        "positive_repeats": POSITIVE_REPEATS,
        "positive_records": len(positive_ids) * POSITIVE_REPEATS,
        "negative_ids": negative_ids,
        "negative_records": len(negative_ids),
        "negative_sampling": "unique rows sorted by SHA256(salt|negative|stage|id)",
        "ordered_record_ids": ordered_record_ids,
        "training_records": len(ordered_record_ids),
        "class_balance": "1:1 by training exposure",
        "ordered_records_sha256": record_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_rows = read_csv(args.data)
    guard_rows = read_csv(args.guard)
    data_by_id = {row["id"]: row for row in data_rows}
    guard_by_id = {row["id"]: row for row in guard_rows}
    if len(data_by_id) != len(data_rows) or len(guard_by_id) != len(guard_rows):
        raise ValueError("ids must be unique")
    if set(data_by_id) != set(guard_by_id):
        raise ValueError("data and connected guard id sets differ")
    for item_id, row in data_by_id.items():
        guard = guard_by_id[item_id]
        if row["category"] != guard["category"] or int(row["label"]) != int(guard["label"]):
            raise ValueError(f"data and guard disagree for id={item_id}")

    stages = {
        **{f"fold_{fold}": build_stage(data_by_id, guard_rows, f"fold_{fold}", fold) for fold in range(5)},
        "full": build_stage(data_by_id, guard_rows, "full", None),
    }
    payload = {
        "experiment_id": "420",
        "immutable": True,
        "created_before_candidate_predictions": True,
        "category": CATEGORY,
        "salt": SALT,
        "selection_uses_predictions": False,
        "selection_uses_error_cohorts": False,
        "selection_uses_labels_only_for_class_balance": True,
        "input_sha256": {
            "data": sha256_file(args.data),
            "connected_guard_rows": sha256_file(args.guard),
        },
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "stages": {name: stage["training_records"] for name, stage in stages.items()},
        "sha256": sha256_file(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
