from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.weight = np.ones(size, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.weight[left] < self.weight[right]:
            left, right = right, left
        self.parent[right] = left
        self.weight[left] += self.weight[right]


def union_by_key(dsu: DisjointSet, keys: list[object]) -> None:
    first: dict[object, int] = {}
    for index, key in enumerate(keys):
        if key in first:
            dsu.union(index, first[key])
        else:
            first[key] = index


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_hash(category: str, ids: list[str]) -> str:
    payload = category + "\n" + "\n".join(sorted(ids))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--first-image-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-version", default="connected_family_guard_v2")
    parser.add_argument("--dataset-version", default="competition_train_v1")
    parser.add_argument("--basket-fold", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows_path = args.output_dir / "rows.csv"
    basket_path = args.output_dir / "basket.csv"
    manifest_path = args.output_dir / "manifest.json"
    existing = [path for path in (rows_path, basket_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite immutable connected guard; choose a new version: "
            + ", ".join(map(str, existing))
        )

    frame = pd.read_csv(args.folds, dtype={"id": str, "group_hash": str})
    archive = np.load(args.first_image_embeddings, allow_pickle=False)
    ids = archive["ids"].astype(str)
    if not np.array_equal(frame.id.to_numpy(), ids):
        raise ValueError("first-image embeddings are not aligned with fold rows")
    if not np.array_equal(frame.label.to_numpy(np.int8), archive["labels"].astype(np.int8)):
        raise ValueError("embedding labels do not match fold rows")
    if not np.array_equal(frame.category.astype(str).to_numpy(), archive["categories"].astype(str)):
        raise ValueError("embedding categories do not match fold rows")
    image_embeddings = archive["embeddings"]

    component_ids = np.empty(len(frame), dtype=object)
    component_fold_counts = np.empty(len(frame), dtype=np.int8)
    component_sizes = np.empty(len(frame), dtype=np.int32)
    source_text_duplicate = np.zeros(len(frame), dtype=bool)
    source_image_duplicate = np.zeros(len(frame), dtype=bool)
    category_summary: dict[str, dict[str, int]] = {}
    for category in sorted(frame.category.unique()):
        positions = np.flatnonzero(frame.category.to_numpy() == category)
        local = frame.iloc[positions]
        dsu = DisjointSet(len(positions))
        text_keys = local.group_hash.astype(str).tolist()
        image_keys = [image_embeddings[position].tobytes() for position in positions]
        text_counts = pd.Series(text_keys).value_counts()
        image_counts = pd.Series(image_keys).value_counts()
        source_text_duplicate[positions] = [text_counts[key] > 1 for key in text_keys]
        source_image_duplicate[positions] = [image_counts[key] > 1 for key in image_keys]
        union_by_key(dsu, text_keys)
        union_by_key(dsu, image_keys)
        roots = np.asarray([dsu.find(index) for index in range(len(positions))])
        local_component_count = 0
        local_crossing_count = 0
        local_crossing_rows = 0
        for root in np.unique(roots):
            members_local = np.flatnonzero(roots == root)
            members = positions[members_local]
            folds = sorted(set(frame.fold.iloc[members].astype(int)))
            value = component_hash(category, frame.id.iloc[members].astype(str).tolist())
            component_ids[members] = value
            component_fold_counts[members] = len(folds)
            component_sizes[members] = len(members)
            local_component_count += 1
            if len(folds) > 1:
                local_crossing_count += 1
                local_crossing_rows += len(members)
        category_summary[category] = {
            "rows": int(len(positions)),
            "components": int(local_component_count),
            "cross_fold_components": int(local_crossing_count),
            "unsafe_rows": int(local_crossing_rows),
            "safe_rows": int(len(positions) - local_crossing_rows),
            "rows_in_exact_text_duplicates": int(source_text_duplicate[positions].sum()),
            "rows_in_exact_image_duplicates": int(source_image_duplicate[positions].sum()),
        }

    output = frame.copy()
    output["connected_component"] = component_ids.astype(str)
    output["component_size"] = component_sizes
    output["component_fold_count"] = component_fold_counts
    output["exact_text_duplicate"] = source_text_duplicate
    output["exact_first_image_duplicate"] = source_image_duplicate
    output["safe_for_selection"] = component_fold_counts == 1
    crossing = output.groupby(["category", "connected_component"]).fold.nunique()
    safe_crossing = output[output.safe_for_selection].groupby(
        ["category", "connected_component"]
    ).fold.nunique()
    invariant = {
        "all_cross_fold_components_flagged_unsafe": bool(
            (output.component_fold_count == output.groupby(
                ["category", "connected_component"]
            ).fold.transform("nunique")).all()
        ),
        "safe_component_max_fold_count": int(safe_crossing.max()),
        "safe_components_crossing_folds": int((safe_crossing > 1).sum()),
        "all_ids_unique": bool(not output.id.duplicated().any()),
        "all_rows_accounted_for": bool(len(output) == len(frame)),
    }
    if not all(
        [
            invariant["all_cross_fold_components_flagged_unsafe"],
            invariant["safe_component_max_fold_count"] == 1,
            invariant["safe_components_crossing_folds"] == 0,
            invariant["all_ids_unique"],
            invariant["all_rows_accounted_for"],
        ]
    ):
        raise ValueError(f"connected guard invariant failed: {invariant}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(rows_path, index=False)
    output[(output.fold == args.basket_fold) & output.safe_for_selection].to_csv(
        basket_path, index=False
    )
    manifest = {
        "evaluation_version": args.evaluation_version,
        "dataset_version": args.dataset_version,
        "immutable": True,
        "protocol": "Historical grouped_text_v1 folds with a sealed selection mask excluding every category-specific connected component that spans folds; edges are exact normalized full text or exact first-image fp16 embedding equality",
        "base_folds": str(args.folds),
        "edge_types": ["exact_normalized_full_text", "exact_first_image_fp16_embedding"],
        "rows": int(len(output)),
        "safe_rows": int(output.safe_for_selection.sum()),
        "unsafe_rows": int((~output.safe_for_selection).sum()),
        "components": int(len(crossing)),
        "cross_fold_components": int((crossing > 1).sum()),
        "category_summary": category_summary,
        "basket_fold": args.basket_fold,
        "safe_basket_rows": int(((output.fold == args.basket_fold) & output.safe_for_selection).sum()),
        "invariants": invariant,
        "input_sha256": {
            "folds": sha256(args.folds),
            "first_image_embeddings": sha256(args.first_image_embeddings),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["output_sha256"] = {
        "rows": sha256(rows_path),
        "basket": sha256(basket_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
