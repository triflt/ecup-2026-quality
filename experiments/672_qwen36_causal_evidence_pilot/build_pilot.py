#!/usr/bin/env python3
"""Freeze a label-blind 40-row multimodal explanation pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "frozen_spec.json").read_text(encoding="utf-8"))
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
SCOPE_CUES = re.compile(
    r"(?i)\b(?:горелк\w*|плит\w*|ламп\w*|обогревател\w*|зажигалк\w*|"
    r"баллон\w*\s+(?:не\s+)?входит|без\s+(?:газа|баллона|топлива)|"
    r"приобрета\w*\s+отдельно|совместим\w*\s+с|работает\s+от|"
    r"в\s+комплект\w*\s+(?:входит|есть)|комплект\w*\s+с)\b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def clean(value: object, limit: int) -> str:
    text = "" if pd.isna(value) else html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head].rstrip() + " … " + text[-(limit - head) :].lstrip()


def discover_ids(paths: list[Path]) -> tuple[set[str], dict[str, str]]:
    ids: set[str] = set()
    hashes: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("id", "row_id", "product_id"):
                candidate = value.get(key)
                if candidate is not None and str(candidate).strip():
                    ids.add(str(candidate).strip())
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                visit(list(csv.DictReader(stream)))
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        visit(json.loads(line))
        else:
            visit(json.loads(path.read_text(encoding="utf-8")))
        hashes[str(path)] = sha256_file(path)
    return ids, hashes


def load_baseline(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != {"id", "prediction", "source"}:
        raise ValueError("exact-140 baseline schema mismatch")
    result: dict[str, int] = {}
    for row in rows:
        prediction = int(row["prediction"])
        if prediction not in (0, 1) or row["source"] != "exact_full140_semantic_v3":
            raise ValueError("exact-140 baseline contract mismatch")
        if row["id"] in result:
            raise ValueError("duplicate baseline ID")
        result[row["id"]] = prediction
    return result


def _pick_stratum(rows: list[dict[str, Any]], category: str, prediction: int) -> list[dict[str, Any]]:
    target = SPEC["rows_per_category_verdict"]
    candidates = [
        row for row in rows if row["category"] == category and row["frozen_prediction"] == prediction
    ]
    candidates.sort(
        key=lambda row: (
            0
            if category == "Легковоспламеняющиеся" and row["scope_priority"]
            else 1,
            stable_hash(SPEC["namespace"], category, prediction, row["component"], row["id"]),
        )
    )
    if len(candidates) < target:
        raise ValueError(f"only {len(candidates)} rows in stratum {(category, prediction)}")
    return candidates[:target]


def build(
    *,
    data: Path,
    folds: Path,
    baseline_predictions: Path,
    images_zip: Path,
    exclusion_manifest: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")

    # `label` is deliberately absent from both usecols lists.
    features = pd.read_csv(data, usecols=["id", "category", "name", "description"], dtype={"id": str})
    membership = pd.read_csv(
        folds,
        usecols=["id", "category", "semantic_component", "split", "development_fold"],
        dtype={"id": str},
    )
    development = membership.loc[membership["split"].astype(str) == "development"].copy()
    if len(development) != 11118 or development["development_fold"].isna().any():
        raise ValueError("semantic-v3 development scope mismatch")
    merged = development.merge(features, on=["id", "category"], how="left", validate="one_to_one")
    if merged[["name", "description"]].isna().all(axis=1).any():
        raise ValueError("development rows do not map to source features")

    baseline = load_baseline(baseline_predictions)
    if set(development["id"]) != set(baseline):
        raise ValueError("baseline must exactly cover semantic-v3 development rows")
    excluded_ids, exclusion_hashes = discover_ids(exclusion_manifest)
    excluded_components = set(
        development.loc[development["id"].isin(excluded_ids), "semantic_component"].astype(str)
    )

    with zipfile.ZipFile(images_zip) as archive:
        available = set(archive.namelist())

    candidates: list[dict[str, Any]] = []
    seen_components: set[str] = set()
    merged = merged.sort_values(
        "id", key=lambda values: values.map(lambda value: stable_hash(SPEC["namespace"], value))
    )
    for source in merged.to_dict("records"):
        row_id = str(source["id"])
        component = str(source["semantic_component"])
        image_member = f"images/{row_id}/0.jpg"
        if component in excluded_components or component in seen_components or image_member not in available:
            continue
        name = clean(source["name"], 320)
        description = clean(source["description"], 1800)
        seen_components.add(component)
        candidates.append(
            {
                "id": row_id,
                "component": component,
                "fold": int(source["development_fold"]),
                "category": str(source["category"]),
                "name": name,
                "description": description,
                "frozen_prediction": baseline[row_id],
                "image_member": image_member,
                "scope_priority": bool(SCOPE_CUES.search(f"{name} {description}")),
            }
        )

    selected: list[dict[str, Any]] = []
    for category in CATEGORIES:
        for prediction in (0, 1):
            selected.extend(_pick_stratum(candidates, category, prediction))
    selected.sort(key=lambda row: stable_hash(SPEC["namespace"], "order", row["component"], row["id"]))
    if len(selected) != SPEC["rows"] or len({row["component"] for row in selected}) != SPEC["rows"]:
        raise AssertionError("pilot must contain 40 fresh semantic components")

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / "pilot_runtime.jsonl"
    with runtime_path.open("x", encoding="utf-8") as stream:
        for global_index, row in enumerate(selected):
            record = {
                "schema_version": "exp672_runtime_row_v1",
                "global_index": global_index,
                "id": row["id"],
                "fold": row["fold"],
                "semantic_component": row["component"],
                "category": row["category"],
                "name": row["name"],
                "description": row["description"],
                "frozen_prediction": row["frozen_prediction"],
                "image_member": row["image_member"],
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema_version": "exp672_private_manifest_v1",
        "experiment_id": "672",
        "namespace": SPEC["namespace"],
        "rows": len(selected),
        "unique_components": len({row["component"] for row in selected}),
        "labels_read": 0,
        "sealed_rows_read": 0,
        "public_used": False,
        "baseline_source": "exact_full140_semantic_v3",
        "strata": dict(Counter(f"{row['category']}|{row['frozen_prediction']}" for row in selected)),
        "folds": dict(Counter(str(row["fold"]) for row in selected)),
        "scope_priority_rows": sum(row["scope_priority"] for row in selected),
        "excluded_row_ids_discovered": len(excluded_ids),
        "excluded_semantic_components": len(excluded_components),
        "runtime_sha256": sha256_file(runtime_path),
        "source_sha256": {
            "data": sha256_file(data),
            "folds": sha256_file(folds),
            "baseline_predictions": sha256_file(baseline_predictions),
            "images_zip": sha256_file(images_zip),
            "frozen_spec": sha256_file(HERE / "frozen_spec.json"),
            "exclusions": exclusion_hashes,
        },
        "ordered_row_ids": [row["id"] for row in selected],
        "ordered_components": [row["component"] for row in selected],
    }
    manifest_path = output_dir / "private_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in manifest.items() if not key.startswith("ordered_")}
    summary["private_manifest_sha256"] = sha256_file(manifest_path)
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    result = build(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
