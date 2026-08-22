from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from fuzzy_annotator_prior_cv import (
    compose_text,
    evaluate_fixed_oof,
    fingerprint,
    normalize,
)
from shingle_neighbor_prior_cv import canonical, f1, stable_shingle_hash


ROOT = Path("research")
DATA = ROOT / "data.csv"
FUSION = ROOT / "qwen3vl-qwen35-5fold-nested-fusion.npz"
OUTPUT = ROOT / "qwen3vl-qwen35-shingle-prior-index.npz"
REPORT = ROOT / "qwen3vl-qwen35-shingle-prior-index-report.json"
CATEGORY = "БАД"
CONFIG = (0.95, 2, 3, 0.999)


def row_shingles(name: object, description: object) -> np.ndarray:
    tokens = canonical(f"{name} {description}", mask_digits=True).split()
    values = {
        stable_shingle_hash(CATEGORY, tokens[start:start + 5])
        for start in range(max(0, len(tokens) - 4))
    }
    return np.asarray(sorted(values), dtype=np.uint64)


def build_index(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    rows = [row_shingles(row.name, row.description) for row in frame.itertuples(index=False)]
    frequencies: Counter[int] = Counter()
    for values in rows:
        frequencies.update(map(int, values))
    useful = np.asarray(
        sorted(key for key, count in frequencies.items() if 2 <= count <= 30),
        dtype=np.uint64,
    )
    useful_set = set(map(int, useful))
    posting_lists: dict[int, list[int]] = defaultdict(list)
    for row_index, values in enumerate(rows):
        for value in values:
            integer = int(value)
            if integer in useful_set:
                posting_lists[integer].append(row_index)
    indptr = np.zeros(len(useful) + 1, dtype=np.int64)
    postings = []
    for index, value in enumerate(useful):
        postings.extend(posting_lists[int(value)])
        indptr[index + 1] = len(postings)
    arrays = {
        "version": np.asarray([1], dtype=np.int16),
        "category": np.asarray([CATEGORY]),
        "similarity_threshold": np.asarray([CONFIG[0]], dtype=np.float32),
        "minimum_neighbors": np.asarray([CONFIG[1]], dtype=np.int16),
        "top_k": np.asarray([CONFIG[2]], dtype=np.int16),
        "confidence_min": np.asarray([CONFIG[3]], dtype=np.float32),
        "unique_shingles": useful,
        "posting_indptr": indptr,
        "posting_rows": np.asarray(postings, dtype=np.int32),
        "row_shingle_counts": np.asarray([len(values) for values in rows], dtype=np.int32),
        "row_labels": frame.label.to_numpy(dtype=np.int8),
    }
    return arrays, rows


def candidate_rows(
    target_shingles: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> list[tuple[float, float, int, int]]:
    unique = arrays["unique_shingles"]
    indptr = arrays["posting_indptr"]
    posting_rows = arrays["posting_rows"]
    positions = np.searchsorted(unique, target_shingles)
    valid = positions < len(unique)
    positions = positions[valid]
    values = target_shingles[valid]
    positions = positions[unique[positions] == values]
    counts: Counter[int] = Counter()
    for position in positions:
        start, end = int(indptr[position]), int(indptr[position + 1])
        counts.update(map(int, posting_rows[start:end]))
    result = []
    target_size = len(target_shingles)
    row_sizes = arrays["row_shingle_counts"]
    for candidate, intersection in counts.items():
        if intersection < 8:
            continue
        candidate_size = int(row_sizes[candidate])
        minimum_size = min(target_size, candidate_size)
        union_size = target_size + candidate_size - intersection
        if not minimum_size or not union_size:
            continue
        containment = intersection / minimum_size
        jaccard = intersection / union_size
        if containment >= 0.40 and jaccard >= 0.20:
            result.append((containment, jaccard, intersection, candidate))
    result.sort(reverse=True)
    return result[:50]


def validate_oof(
    full_frame: pd.DataFrame,
    category_frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    rows: list[np.ndarray],
    fusion: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    base = fusion["full_oof_predictions"].astype(np.int8)
    _, _, baseline = evaluate_fixed_oof(
        full_frame, base, CATEGORY, ("canonical_text_mask_digits", 1, 0.999)
    )
    predictions = baseline.copy()
    global_positions = category_frame["global_position"].to_numpy(dtype=np.int32)
    local_folds = category_frame.fold.to_numpy(dtype=np.int8)
    labels = full_frame.label.to_numpy(dtype=np.int8)
    threshold, minimum_neighbors, top_k, confidence_min = CONFIG
    hits = changed = 0
    for target_local, target_global in enumerate(global_positions):
        selected = []
        for containment, _, _, candidate_local in candidate_rows(rows[target_local], arrays):
            if containment < threshold or local_folds[candidate_local] == local_folds[target_local]:
                continue
            selected.append(candidate_local)
            if len(selected) >= top_k:
                break
        if len(selected) < minimum_neighbors:
            continue
        donor_labels = arrays["row_labels"][selected]
        mean = float(donor_labels.mean())
        confidence = max(mean, 1 - mean)
        if confidence < confidence_min or mean == 0.5:
            continue
        prediction = int(mean >= 0.5)
        hits += 1
        changed += int(prediction != predictions[target_global])
        predictions[target_global] = prediction
    category_positions = np.flatnonzero(full_frame.category.to_numpy() == CATEGORY)
    return {
        "category": CATEGORY,
        "config": list(CONFIG),
        "oof_f1": f1(labels[category_positions], predictions[category_positions]),
        "expected_oof_f1": 0.9572418759564317,
        "hits": hits,
        "changed": changed,
        "matches_expected": abs(
            f1(labels[category_positions], predictions[category_positions])
            - 0.9572418759564317
        ) < 1e-12,
    }


def main() -> None:
    full = pd.read_csv(DATA)
    full["name"] = full.name.fillna("").astype(str)
    full["description"] = full.description.fillna("").astype(str)
    full["normalized_name"] = full.name.map(normalize)
    texts = [
        compose_text(name, description)
        for name, description in zip(full.name, full.description)
    ]
    full["text_hash"] = [fingerprint(text) for text in texts]
    full["canonical_text_mask_digits"] = [
        canonical(text, mask_digits=True) for text in texts
    ]
    fusion = np.load(FUSION, allow_pickle=True)
    full["fold"] = fusion["folds"].astype(np.int8)
    category = full[full.category == CATEGORY].copy()
    category["global_position"] = category.index.to_numpy(dtype=np.int32)
    category.reset_index(drop=True, inplace=True)
    arrays, rows = build_index(category)
    np.savez_compressed(OUTPUT, **arrays)
    validation = validate_oof(full, category, arrays, rows, fusion)
    report = {
        "rows": len(category),
        "unique_useful_shingles": len(arrays["unique_shingles"]),
        "posting_entries": len(arrays["posting_rows"]),
        "index_bytes": OUTPUT.stat().st_size,
        "validation": validation,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not validation["matches_expected"]:
        raise RuntimeError("serialized shingle index does not reproduce the selected OOF score")


if __name__ == "__main__":
    main()
