from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite fast teacher runtime")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.full_runtime / "train.jsonl"
    source_audit_path = args.full_runtime / "runtime_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("schema_version") != "exp715_full_runtime_v1":
        raise ValueError("full runtime schema mismatch")
    if source_audit.get("train_sha256") != sha256(source_path):
        raise ValueError("full runtime checksum mismatch")
    rows = []
    with source_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid full runtime JSONL at line {line_number}"
                ) from error
    rows = [row for row in rows if int(row["fold"]) != 3]
    if not rows or any(int(row["fold"]) == 3 for row in rows):
        raise ValueError("fold-3 validation leaked into fast teacher runtime")
    train_path = args.output_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "schema_version": "exp705_teacher_runtime_v1",
        "experiment_id": "697",
        "consumer_experiment_id": "705",
        "fold": 3,
        "outer_validation_excluded": True,
        "source_full_runtime_sha256": sha256(source_audit_path),
        "train_occurrences": len(rows),
        "train_unique_ids": len({str(row["id"]) for row in rows}),
        "train_sha256": sha256(train_path),
        "decision": "DEADLINE_FAST_TEACHER_RUNTIME_FROZEN",
    }
    (args.output_dir / "runtime_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
