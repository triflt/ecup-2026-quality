from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from grid_contract import FULL_FOLDS, sha256_file

AUDIT_SEED = 645
ROWS_PER_CATEGORY_FOLD = 20


def _rank(row_id: str) -> str:
    return hashlib.sha256(f"{AUDIT_SEED}:{row_id}".encode()).hexdigest()


def freeze(*, registry_path: Path, output_path: Path, report_path: Path) -> dict:
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite the frozen audit")
    frame = pd.read_csv(
        registry_path,
        usecols=["id", "category", "semantic_component", "split", "development_fold"],
        dtype={"id": str, "semantic_component": str},
    )
    frame = frame.loc[frame["split"].astype(str).eq("development")].copy()
    if set(frame["development_fold"].astype(int)) != set(FULL_FOLDS):
        raise ValueError("semantic-family development folds are incomplete")
    selected = []
    for (category, fold), local in frame.groupby(["category", "development_fold"], sort=True):
        local = local.copy()
        local["rank"] = local["id"].map(_rank)
        local = local.sort_values(["rank", "id"])
        local = local.drop_duplicates("semantic_component", keep="first")
        if len(local) < ROWS_PER_CATEGORY_FOLD:
            raise ValueError(f"not enough unique components for {category}/fold{fold}")
        selected.append(local.head(ROWS_PER_CATEGORY_FOLD)[["id"]])
    audit = pd.concat(selected, ignore_index=True)
    audit["track"] = "evidence_first_27b"
    if len(audit) != 200 or audit["id"].duplicated().any():
        raise ValueError("frozen audit must contain 200 unique IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)
    report = {
        "schema_version": 1,
        "experiment_id": "645",
        "selection": "sha256-ranked one row per semantic component",
        "seed": AUDIT_SEED,
        "rows_per_category_fold": ROWS_PER_CATEGORY_FOLD,
        "rows": len(audit),
        "labels_read": 0,
        "sealed_rows": 0,
        "track": "evidence_first_27b",
        "registry_sha256": sha256_file(registry_path),
        "audit_sha256": sha256_file(output_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the label-blind 200-row evidence audit.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        registry_path=args.registry,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
