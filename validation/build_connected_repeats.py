from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 31415, 20260822])
    parser.add_argument("--evaluation-version", default="connected_family_repeated_v1")
    parser.add_argument("--dataset-version", default="competition_train_v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("repeat seeds must be unique")
    rows_path = args.output_dir / "rows.csv"
    manifest_path = args.output_dir / "manifest.json"
    existing = [path for path in (rows_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite immutable connected repeats: "
            + ", ".join(map(str, existing))
        )
    frame = pd.read_csv(
        args.guard,
        dtype={"id": str, "group_hash": str, "connected_component": str},
    )
    safe = frame.safe_for_selection.astype(bool).to_numpy()
    output = frame[
        [
            "id",
            "category",
            "label",
            "fold",
            "group_hash",
            "connected_component",
            "safe_for_selection",
        ]
    ].copy()
    summary: dict[str, object] = {}
    for repeat, seed in enumerate(args.seeds):
        column = f"repeat_{repeat}_fold"
        values = np.full(len(frame), -1, dtype=np.int8)
        category_summary: dict[str, object] = {}
        for category in sorted(frame.category.unique()):
            positions = np.flatnonzero(safe & (frame.category.to_numpy() == category))
            labels = frame.label.iloc[positions].to_numpy(np.int8)
            groups = frame.connected_component.iloc[positions].astype(str).to_numpy()
            splitter = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=seed
            )
            for fold, (_, validation) in enumerate(
                splitter.split(np.zeros(len(positions)), labels, groups)
            ):
                values[positions[validation]] = fold
            local = pd.DataFrame(
                {"fold": values[positions], "label": labels, "group": groups}
            )
            component_fold_counts = local.groupby("group").fold.nunique()
            if int(component_fold_counts.max()) != 1:
                raise ValueError(
                    f"connected component crosses repeat={repeat} category={category}"
                )
            category_summary[category] = {
                "rows": int(len(local)),
                "components": int(local.group.nunique()),
                "fold_rows": {
                    str(key): int(value)
                    for key, value in local.fold.value_counts().sort_index().items()
                },
                "fold_positives": {
                    str(key): int(value)
                    for key, value in local.groupby("fold").label.sum().sort_index().items()
                },
                "component_max_fold_count": int(component_fold_counts.max()),
            }
        if not np.all(values[safe] >= 0) or not np.all(values[~safe] == -1):
            raise ValueError(f"repeat {repeat} row accounting failed")
        output[column] = values
        summary[str(repeat)] = {"seed": seed, "categories": category_summary}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(rows_path, index=False)
    manifest = {
        "evaluation_version": args.evaluation_version,
        "dataset_version": args.dataset_version,
        "immutable": True,
        "protocol": "Three predeclared category-specific StratifiedGroupKFold repeats over connected_family_guard_v2 safe components; unsafe rows receive fold -1 and are never selected",
        "guard": str(args.guard),
        "rows_total": int(len(output)),
        "safe_rows": int(safe.sum()),
        "unsafe_rows": int((~safe).sum()),
        "n_splits": 5,
        "seeds": args.seeds,
        "repeats": summary,
        "input_sha256": {"guard": sha256(args.guard)},
        "output_sha256": {"rows": sha256(rows_path)},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
