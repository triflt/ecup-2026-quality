from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
from pathlib import Path


EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"expected one {name}, got {paths}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.data.exists():
        url = os.environ.get("DATA_CSV_URL")
        if not url:
            raise ValueError("DATA_CSV_URL is required")
        args.data.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, args.data)
    if sha256_file(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("data SHA mismatch")
    with args.data.open(encoding="utf-8", newline="") as stream:
        data = {str(row["id"]): row for row in csv.DictReader(stream)}

    smoke_paths = sorted(args.teacher_root.rglob("teacher_smoke20_v5.jsonl"))
    teacher_path = smoke_paths[0] if len(smoke_paths) == 1 else find_one(
        args.teacher_root, "explanation_synth_v5.jsonl"
    )
    records = []
    with teacher_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            item_id = str(record["id"])
            source = data[item_id]
            records.append({
                "id": item_id,
                "category": str(source["category"]),
                "label": int(source["label"]),
                "name": str(source["name"]),
                "description": str(source["description"]),
                "image_count": int(record["image_count"]),
                "image_sha256": list(record["image_sha256"]),
                "accepted": bool(record["accepted"]),
                "normalization_actions": list(record.get("normalization_actions") or []),
                "student_scope_repair": {
                    "original_errors": list(
                        (record.get("student_scope_repair") or {}).get("original_errors") or []
                    ),
                    "attempts": [
                        {
                            "attempt": attempt.get("attempt"),
                            "validation_errors": list(attempt.get("validation_errors") or []),
                        }
                        for attempt in (record.get("student_scope_repair") or {}).get("attempts") or []
                    ],
                } if record.get("student_scope_repair") else None,
                "output": record["parsed_output"],
            })
    if len(records) != 20:
        raise ValueError(f"expected 20 pilot rows, got {len(records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact_preview = [
        {
            "id": row["id"],
            "category": row["category"],
            "label": row["label"],
            "name": row["name"],
            "image_count": row["image_count"],
            "accepted": row["accepted"],
            "normalization_actions": row["normalization_actions"],
            "student_scope_repair": row["student_scope_repair"],
            "output": row["output"],
        }
        for row in records
    ]
    for row in compact_preview:
        print(
            "PREVIEW_ROW=" + json.dumps(row, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    print("PREVIEW_SHA256=" + sha256_file(args.output), flush=True)


if __name__ == "__main__":
    main()
