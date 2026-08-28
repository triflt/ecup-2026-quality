from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    sources = {}
    for fold in range(5):
        root = args.runtime_root / f"fold{fold}"
        validation_path = root / "validation.jsonl"
        audit_path = root / "runtime_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        payload = dict(audit)
        declared = payload.pop("contract_sha256", None)
        if declared != canonical_sha256(payload):
            raise ValueError(f"fold{fold} runtime audit self-hash mismatch")
        validation_sha = sha256(validation_path)
        if (
            audit.get("outer_fold") != fold
            or audit.get("validation_labels_written") != 0
            or audit.get("sealed_rows_written") != 0
            or audit.get("output_sha256", {}).get("validation.jsonl")
            != validation_sha
        ):
            raise ValueError(f"fold{fold} runtime audit mismatch")
        local = [
            json.loads(line)
            for line in validation_path.read_text(encoding="utf-8").splitlines()
        ]
        if any("label" in row for row in local):
            raise ValueError("source validation contains labels")
        if any(int(row["fold"]) != fold for row in local):
            raise ValueError("source validation fold mismatch")
        rows.extend(
            {
                "global_index": int(row["global_index"]),
                "id": str(row["id"]),
                "fold": fold,
                "category": str(row["category"]),
            }
            for row in local
        )
        sources[str(fold)] = {
            "runtime_contract_sha256": declared,
            "runtime_audit_file_sha256": sha256(audit_path),
            "validation_sha256": validation_sha,
            "rows": len(local),
        }
    rows.sort(key=lambda row: row["global_index"])
    if [row["global_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("source global_index coverage mismatch")
    keys = [(row["id"], row["category"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("source id/category key is duplicated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    contract = {
        "schema": "exp699_eval_runtime_map_v1",
        "experiment_id": "699",
        "rows": len(rows),
        "folds": sources,
        "map_sha256": sha256(args.output),
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
    }
    contract["self_sha256"] = canonical_sha256(contract)
    args.contract.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
