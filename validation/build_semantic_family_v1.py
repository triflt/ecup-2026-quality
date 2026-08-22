from __future__ import annotations

"""Draft graph-audit builder; deliberately cannot create a sealed validation.

Independent review found that unconditional exact-name, digit-masked-name and
all-gallery perceptual edges can over-merge unrelated products, while ordinary
StratifiedGroupKFold does not balance the rare flammable-positive stratum. Keep
this implementation only as an auditable starting point. A future immutable
version must implement corroborated edges and a deterministic component
balancing optimizer before removing the explicit ``--draft-only`` guard.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ecup_quality.data.text import canonicalize, normalize_text


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def union_by_key(dsu: DisjointSet, keys: list[str], *, valid: np.ndarray | None = None) -> np.ndarray:
    first: dict[str, int] = {}
    duplicate = np.zeros(len(keys), dtype=bool)
    for index, key in enumerate(keys):
        if valid is not None and not bool(valid[index]):
            continue
        if not key:
            continue
        if key in first:
            other = first[key]
            dsu.union(index, other)
            duplicate[index] = True
            duplicate[other] = True
        else:
            first[key] = index
    return duplicate


def stable_component_hash(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def normalized_full_text(name: object, description: object) -> str:
    return normalize_text(f"{name or ''}\n{description or ''}")


def digit_masked_name(name: object) -> str:
    value = canonicalize(name, mask_digits=True)
    alpha_tokens = [token for token in value.split() if token != "#" and any(ch.isalpha() for ch in token)]
    alpha_chars = sum(sum(ch.isalpha() for ch in token) for token in alpha_tokens)
    if len(alpha_tokens) < 3 or alpha_chars < 10 or "#" not in value.split():
        return ""
    return value


def perceptual_key(image: Any) -> tuple[str, str, float]:
    from PIL import Image, ImageStat

    gray = image.convert("L")
    dh = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    ah = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
    d_bits = (dh[:, 1:] > dh[:, :-1]).reshape(-1)
    a_bits = (ah >= ah.mean()).reshape(-1)
    d_value = sum(int(bit) << index for index, bit in enumerate(d_bits))
    a_value = sum(int(bit) << index for index, bit in enumerate(a_bits))
    ratio = image.width / max(1, image.height)
    # A 2% logarithmic bucket keeps resized copies together while preventing
    # square icons from connecting unrelated wide product photos.
    aspect_bucket = str(round(math.log(max(ratio, 1e-6)) / math.log(1.02)))
    contrast = float(ImageStat.Stat(gray.resize((64, 64))).stddev[0])
    return f"{d_value:016x}{a_value:016x}", aspect_bucket, contrast


@dataclass(frozen=True)
class ImageFingerprint:
    exact: str
    perceptual: str
    aspect_bucket: str
    contrast: float


def fingerprint_url(url: str, timeout: int) -> ImageFingerprint:
    from PIL import Image

    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        perceptual, aspect_bucket, contrast = perceptual_key(image)
    return ImageFingerprint(
        exact=hashlib.sha1(payload).hexdigest(),
        perceptual=perceptual,
        aspect_bucket=aspect_bucket,
        contrast=contrast,
    )


def read_image_manifest(path: Path) -> tuple[list[str], list[tuple[str, int, str]]]:
    ids: list[str] = []
    items: list[tuple[str, int, str]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            item_id = str(row["id"])
            ids.append(item_id)
            urls = json.loads(row["image_urls"])
            for image_index, url in enumerate(urls):
                items.append((item_id, image_index, str(url)))
    return ids, items


def write_image_hash_cache(
    *,
    manifest: Path,
    output: Path,
    workers: int,
    timeout: int,
) -> dict[str, object]:
    ids, items = read_image_manifest(manifest)
    by_id: dict[str, list[ImageFingerprint]] = defaultdict(list)
    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fingerprint_url, url, timeout): (item_id, image_index)
            for item_id, image_index, url in items
        }
        for completed, future in enumerate(as_completed(futures), 1):
            item_id, image_index = futures[future]
            try:
                by_id[item_id].append(future.result())
            except Exception as error:  # noqa: BLE001
                # URL is intentionally never persisted or printed.
                failures.append({
                    "id": item_id,
                    "image_index": int(image_index),
                    "error_type": type(error).__name__,
                })
            if completed % 2000 == 0 or completed == len(futures):
                print(
                    f"hashed_images={completed}/{len(futures)} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(
            f"refusing incomplete image hash cache: {len(failures)} downloads failed; "
            f"first={failures[:5]}"
        )
    missing = [item_id for item_id in ids if not by_id[item_id]]
    if missing:
        raise RuntimeError(f"products without image hashes: {missing[:20]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        for item_id in ids:
            fingerprints = sorted(
                by_id[item_id],
                key=lambda item: (item.exact, item.perceptual, item.aspect_bucket),
            )
            row = {
                "id": item_id,
                "images": [
                    {
                        "exact": item.exact,
                        "perceptual": item.perceptual,
                        "aspect_bucket": item.aspect_bucket,
                        "contrast": round(item.contrast, 6),
                    }
                    for item in fingerprints
                ],
            }
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "products": len(ids),
        "images": len(items),
        "download_failures": 0,
        "sha256": sha256(output),
    }


def load_image_hash_cache(path: Path, expected_ids: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    ids = np.asarray([str(row["id"]) for row in rows])
    if not np.array_equal(ids, expected_ids.astype(str)):
        raise ValueError("image hash cache ids are not aligned with data")
    if any(not row["images"] for row in rows):
        raise ValueError("image hash cache contains a product without images")
    return rows


def image_row_keys(
    rows: list[dict[str, object]], *, contrast_min: float
) -> tuple[list[list[str]], list[list[str]]]:
    exact: list[list[str]] = []
    perceptual: list[list[str]] = []
    for row in rows:
        images = row["images"]
        exact.append(sorted({str(image["exact"]) for image in images}))
        perceptual.append(sorted({
            f"{image['perceptual']}:{image['aspect_bucket']}"
            for image in images
            if float(image["contrast"]) >= contrast_min
        }))
    return exact, perceptual


def union_rows_by_tokens(
    dsu: DisjointSet, row_tokens: list[list[str]]
) -> np.ndarray:
    owner: dict[str, int] = {}
    duplicate = np.zeros(len(row_tokens), dtype=bool)
    for index, tokens in enumerate(row_tokens):
        for token in tokens:
            if token in owner:
                other = owner[token]
                dsu.union(index, other)
                duplicate[index] = True
                duplicate[other] = True
            else:
                owner[token] = index
    return duplicate


def assign_splits(
    *,
    frame: pd.DataFrame,
    groups: np.ndarray,
    holdout_splits: int,
    holdout_seed: int,
    dev_splits: int,
    dev_seed: int,
    chosen_holdout_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    strata = (frame["category"].astype(str) + "|" + frame["label"].astype(str)).to_numpy()
    outer = StratifiedGroupKFold(
        n_splits=holdout_splits,
        shuffle=True,
        random_state=holdout_seed,
    )
    holdout = np.zeros(len(frame), dtype=bool)
    for fold, (_, valid) in enumerate(outer.split(np.zeros(len(frame)), strata, groups)):
        if fold == chosen_holdout_fold:
            holdout[valid] = True
    dev_fold = np.full(len(frame), -1, dtype=np.int8)
    dev_positions = np.flatnonzero(~holdout)
    dev_splitter = StratifiedGroupKFold(
        n_splits=dev_splits,
        shuffle=True,
        random_state=dev_seed,
    )
    for fold, (_, valid_local) in enumerate(
        dev_splitter.split(
            np.zeros(len(dev_positions)),
            strata[dev_positions],
            groups[dev_positions],
        )
    ):
        dev_fold[dev_positions[valid_local]] = fold
    if np.any(dev_fold[holdout] != -1) or np.any(dev_fold[~holdout] < 0):
        raise ValueError("incomplete dev/holdout split assignment")
    return holdout, dev_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--image-hash-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--perceptual-contrast-min", type=float, default=8.0)
    parser.add_argument("--holdout-splits", type=int, default=7)
    parser.add_argument("--holdout-seed", type=int, default=20260822)
    parser.add_argument("--dev-splits", type=int, default=5)
    parser.add_argument("--dev-seed", type=int, default=314159)
    parser.add_argument("--evaluation-version", default="semantic_family_graph_audit_draft_v0")
    parser.add_argument("--dataset-version", default="competition_train_v1")
    parser.add_argument(
        "--draft-only",
        action="store_true",
        help="Acknowledge that output is a non-sealed graph audit and cannot score candidates.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.draft_only:
        raise SystemExit(
            "fail-closed: this draft topology is not safe to seal; read "
            "research/agent_notes/semantic_split_design_critique.md and pass "
            "--draft-only only for label-blind graph diagnostics"
        )

    output_paths = [
        args.output_dir / "rows.csv",
        args.output_dir / "dev_folds.csv",
        args.output_dir / "sealed_holdout_ids.csv",
        args.output_dir / "manifest.json",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite immutable validation: " + ", ".join(map(str, existing))
        )

    frame = pd.read_csv(args.data, dtype={"id": str})
    required = {"id", "name", "description", "category", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing data columns: {sorted(missing)}")
    if frame.id.duplicated().any():
        raise ValueError("data ids are not unique")
    manifest_ids, manifest_items = read_image_manifest(args.image_manifest)
    if manifest_ids != frame.id.astype(str).tolist():
        raise ValueError("image manifest ids are not aligned with data")

    if not args.image_hash_cache.exists():
        cache_report = write_image_hash_cache(
            manifest=args.image_manifest,
            output=args.image_hash_cache,
            workers=args.workers,
            timeout=args.download_timeout,
        )
    else:
        cache_report = {
            "products": len(frame),
            "images": len(manifest_items),
            "download_failures": 0,
            "sha256": sha256(args.image_hash_cache),
            "reused": True,
        }
    image_rows = load_image_hash_cache(
        args.image_hash_cache, frame.id.astype(str).to_numpy()
    )

    names = frame.name.fillna("").astype(str)
    descriptions = frame.description.fillna("").astype(str)
    full_text_keys = [
        normalized_full_text(name, description)
        for name, description in zip(names, descriptions)
    ]
    name_keys = [normalize_text(name) for name in names]
    masked_name_keys = [digit_masked_name(name) for name in names]
    exact_image_tokens, perceptual_image_tokens = image_row_keys(
        image_rows, contrast_min=args.perceptual_contrast_min
    )

    dsu = DisjointSet(len(frame))
    edge_flags = {
        "exact_full_text": union_by_key(dsu, full_text_keys),
        "exact_name": union_by_key(dsu, name_keys),
        "digit_masked_name": union_by_key(dsu, masked_name_keys),
        "exact_gallery_image": union_rows_by_tokens(dsu, exact_image_tokens),
        "perceptual_gallery_image": union_rows_by_tokens(dsu, perceptual_image_tokens),
    }
    roots = np.asarray([dsu.find(index) for index in range(len(frame))], dtype=np.int32)
    components = np.empty(len(frame), dtype=object)
    component_sizes = np.empty(len(frame), dtype=np.int32)
    component_category_counts = np.empty(len(frame), dtype=np.int8)
    component_label_counts = np.empty(len(frame), dtype=np.int8)
    for root in np.unique(roots):
        positions = np.flatnonzero(roots == root)
        component = stable_component_hash(frame.id.iloc[positions].astype(str).tolist())
        components[positions] = component
        component_sizes[positions] = len(positions)
        component_category_counts[positions] = frame.category.iloc[positions].nunique()
        component_label_counts[positions] = frame.label.iloc[positions].nunique()

    input_digest = hashlib.sha256(
        (sha256(args.data) + sha256(args.image_hash_cache)).encode("ascii")
    ).hexdigest()
    chosen_holdout_fold = int(input_digest, 16) % args.holdout_splits
    holdout, dev_fold = assign_splits(
        frame=frame,
        groups=components.astype(str),
        holdout_splits=args.holdout_splits,
        holdout_seed=args.holdout_seed,
        dev_splits=args.dev_splits,
        dev_seed=args.dev_seed,
        chosen_holdout_fold=chosen_holdout_fold,
    )

    output = frame[["id", "category", "label"]].copy()
    output["semantic_component"] = components.astype(str)
    output["component_size"] = component_sizes
    output["component_category_count"] = component_category_counts
    output["component_label_count"] = component_label_counts
    for name, flags in edge_flags.items():
        output[f"in_{name}_duplicate"] = flags
    output["sealed_holdout"] = holdout
    output["dev_fold"] = dev_fold

    component_frame = output.groupby("semantic_component", sort=False).agg(
        rows=("id", "size"),
        holdout_values=("sealed_holdout", "nunique"),
        dev_fold_values=("dev_fold", lambda values: values[values >= 0].nunique()),
    )
    if int(component_frame.holdout_values.max()) != 1:
        raise ValueError("component crosses the sealed holdout boundary")
    if int(component_frame.dev_fold_values.max()) > 1:
        raise ValueError("component crosses development folds")

    stratum = output.category.astype(str) + "|" + output.label.astype(str)
    stratum_summary: dict[str, object] = {}
    for value in sorted(stratum.unique()):
        local = stratum == value
        stratum_summary[value] = {
            "rows": int(local.sum()),
            "components": int(output.loc[local, "semantic_component"].nunique()),
            "sealed_holdout_rows": int((local & holdout).sum()),
            "development_rows": int((local & ~holdout).sum()),
            "development_fold_rows": {
                str(fold): int((local & (dev_fold == fold)).sum())
                for fold in range(args.dev_splits)
            },
        }
    component_counts = pd.Series(components.astype(str)).value_counts()
    edge_summary = {
        name: {
            "rows_in_duplicate_edge": int(flags.sum()),
            "duplicate_rate": float(flags.mean()),
        }
        for name, flags in edge_flags.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path, dev_path, holdout_path, manifest_path = output_paths
    output.to_csv(rows_path, index=False)
    output.loc[~holdout].to_csv(dev_path, index=False)
    output.loc[holdout, ["id", "semantic_component"]].to_csv(holdout_path, index=False)
    manifest = {
        "evaluation_version": args.evaluation_version,
        "dataset_version": args.dataset_version,
        "immutable": False,
        "status": "draft_graph_audit_not_valid_for_candidate_scoring",
        "created_at": "2026-08-22",
        "protocol": (
            "Cross-category label-blind connected components over conservative text and "
            "all-gallery image identities; one deterministically selected 1/7 component "
            "fold is sealed before candidate scoring, remaining components form five dev folds"
        ),
        "labels_used_only_for_stratification_and_audit": True,
        "candidate_scores_used": False,
        "known_blockers": [
            "generic exact-name and digit-masked-name keys need corroboration",
            "auxiliary gallery images need position-aware corroboration",
            "perceptual image hashes must never form an edge alone",
            "ordinary StratifiedGroupKFold must be replaced by a four-stratum component optimizer",
            "the full baseline must be retrained after excluding the eventual sealed components",
        ],
        "edge_types": [
            "exact_normalized_full_text",
            "exact_normalized_name",
            "digit_masked_name_with_minimum_three_alpha_tokens_and_numeric_marker",
            "exact_sha1_of_any_gallery_image",
            "exact_128bit_dhash_ahash_with_2pct_aspect_bucket_and_contrast_guard",
        ],
        "perceptual_contrast_min": args.perceptual_contrast_min,
        "rows": len(output),
        "components": int(component_counts.size),
        "duplicate_components": int((component_counts > 1).sum()),
        "rows_in_duplicate_components": int(component_counts[component_counts > 1].sum()),
        "largest_component": int(component_counts.max()),
        "mixed_category_components": int(
            output.groupby("semantic_component").category.nunique().gt(1).sum()
        ),
        "mixed_label_components": int(
            output.groupby("semantic_component").label.nunique().gt(1).sum()
        ),
        "sealed_holdout": {
            "selection": "sha256(data_sha256 + image_hash_cache_sha256) modulo 7",
            "n_splits": args.holdout_splits,
            "seed": args.holdout_seed,
            "chosen_fold": chosen_holdout_fold,
            "rows": int(holdout.sum()),
            "components": int(output.loc[holdout, "semantic_component"].nunique()),
            "must_not_be_scored_until_recipe_preregistered": True,
        },
        "development": {
            "n_splits": args.dev_splits,
            "seed": args.dev_seed,
            "rows": int((~holdout).sum()),
            "components": int(output.loc[~holdout, "semantic_component"].nunique()),
        },
        "strata": stratum_summary,
        "edges": edge_summary,
        "image_hash_cache": cache_report,
        "invariants": {
            "all_ids_unique": bool(not output.id.duplicated().any()),
            "all_rows_accounted_for": bool(len(output) == len(frame)),
            "components_crossing_holdout_boundary": int(
                (component_frame.holdout_values > 1).sum()
            ),
            "components_crossing_dev_folds": int(
                (component_frame.dev_fold_values > 1).sum()
            ),
            "holdout_rows_have_dev_fold_minus_one": bool(np.all(dev_fold[holdout] == -1)),
            "development_rows_have_one_fold": bool(np.all(dev_fold[~holdout] >= 0)),
            "manifest_ids_match_data": manifest_ids == frame.id.astype(str).tolist(),
        },
        "input_sha256": {
            "data": sha256(args.data),
            "image_manifest": sha256(args.image_manifest),
            "image_hash_cache": sha256(args.image_hash_cache),
        },
        "output_sha256": {
            "rows": sha256(rows_path),
            "dev_folds": sha256(dev_path),
            "sealed_holdout_ids": sha256(holdout_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
