from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pair_selector import build_pair_plan
from parent_selector import select_parent_training_records

FROZEN_GUARD_SHA256 = "b2cd736226c05122f33aa0c898b31408ef469df0cfe7fe739a95c5b5b94a526a"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen exp540 donor pair manifest.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=(0, 3))
    parser.add_argument("--seed", default=42, type=int, choices=(42,))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path, oof_path, guard_path = (
        args.data.resolve(),
        args.oof.resolve(),
        args.guard.resolve(),
    )
    guard_sha256 = file_sha256(guard_path)
    if guard_sha256 != FROZEN_GUARD_SHA256:
        raise ValueError("connected family guard is not the frozen v2 artifact")
    frame = pd.read_csv(data_path, dtype={"id": str})
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    guard = pd.read_csv(guard_path, dtype={"id": str, "connected_component": str})
    oof = np.load(oof_path, allow_pickle=True)
    if not np.array_equal(frame.id.astype(str).to_numpy(), oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    records, parent_audit = select_parent_training_records(
        frame, oof, seed=args.seed, holdout_fold=args.fold
    )
    _, manifest, audit = build_pair_plan(
        frame,
        guard,
        oof["fold_ids"].astype(np.int8),
        records,
        holdout_fold=args.fold,
    )
    audit["seed"] = args.seed
    audit["parent_selection"] = parent_audit
    audit["input_sha256"] = {
        "data": file_sha256(data_path),
        "oof": file_sha256(oof_path),
        "connected_guard": guard_sha256,
    }
    output = args.output_dir.resolve()
    manifest_path = output / "pair_manifest.jsonl"
    audit_path = output / "pair_audit.json"
    existing = [path for path in (manifest_path, audit_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite frozen pair artifacts: " + ", ".join(map(str, existing))
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": audit["decision"],
                "fold": args.fold,
                "pair_batches": audit["pair_batches"],
                "realized_pairs": audit["realized_pairs"],
                "same_family_pairs": audit["same_family_pairs"],
                "nearest_family_pairs": audit["nearest_family_pairs"],
                "audit": str(audit_path),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if audit["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
