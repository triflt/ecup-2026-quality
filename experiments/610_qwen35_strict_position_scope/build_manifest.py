from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from position_protocol import SCREEN_FOLDS, build_augmentation_plan
from position_shared import load_dependency_light_selector, load_exp600_dependencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the strict-v2 position manifest.")
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=SCREEN_FOLDS)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a manifest directory")
    protocol, _ = load_exp600_dependencies()
    selector = load_dependency_light_selector()
    with tempfile.TemporaryDirectory(prefix="exp610-protocol-") as temporary:
        prepared = protocol.materialize_runtime_fold(
            runtime_dir=args.runtime_dir.resolve(),
            output_dir=Path(temporary) / "protocol_inputs",
            component="specialist",
            outer_fold=args.fold,
        )
        frame = pd.read_csv(prepared["data"], dtype={"id": str}).fillna("")
        oof = np.load(prepared["selector"], allow_pickle=False)
        records, parent_audit = selector.select_parent_training_records(
            frame,
            oof,
            seed=42,
            holdout_fold=args.fold,
            full_train=False,
        )
        _, manifest, audit = build_augmentation_plan(
            frame,
            records,
            oof["fold_ids"].astype(np.int8),
            outer_fold=args.fold,
        )
    audit["parent_selection"] = parent_audit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "augmentation_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    audit_path = args.output_dir / "augmentation_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"audit": str(audit_path), "decision": audit["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
