from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pair_selector import build_pair_plan, canonical_sha256
from parent_selector import select_parent_training_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen exp560 parent-intersection pairs.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=(0, 3))
    parser.add_argument("--seed", default=42, type=int, choices=(42,))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.data.resolve(), dtype={"id": str})
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    oof = np.load(args.oof.resolve(), allow_pickle=True)
    if not np.array_equal(frame.id.astype(str).to_numpy(), oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    records, parent_audit = select_parent_training_records(
        frame, oof, seed=args.seed, holdout_fold=args.fold
    )
    _, manifest, audit = build_pair_plan(
        frame,
        oof["fold_ids"].astype(np.int8),
        records,
        holdout_fold=args.fold,
    )
    audit["seed"] = args.seed
    audit["parent_selection"] = parent_audit
    audit.pop("audit_sha256", None)
    audit["audit_sha256"] = canonical_sha256(audit)
    output = args.output_dir.resolve()
    manifest_path = output / "pair_manifest.jsonl"
    audit_path = output / "pair_audit.json"
    existing = [path for path in (manifest_path, audit_path) if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite frozen pair artifacts")
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
                "parent_intersection": audit["eligible_parent_intersection_pairs"],
            },
            sort_keys=True,
        )
    )
    return 0 if audit["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
