from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("research")
DATA = Path(os.environ.get("ECUP_DATA", ROOT / "data.csv"))
FUSION = ROOT / "qwen3vl-qwen35-5fold-nested-fusion.npz"
REPORT = ROOT / "qwen3vl-qwen35-shingle-neighbor-prior-report.json"
AUDIT = ROOT / "qwen3vl-qwen35-shingle-neighbor-audit.tsv"

BASE_CONFIGS = {
    "БАД": (2, 2 / 3, 2, 0.999),
    "Легковоспламеняющиеся": (1, 0.999, 999, 0.999),
}
CANONICAL_CONFIGS = {
    "БАД": ("canonical_text_mask_digits", 1, 0.999),
    "Легковоспламеняющиеся": None,
}


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def compose_text(name: object, description: object) -> str:
    normalized_name = normalize(name)
    return f"{normalized_name}\n{normalized_name}\n{normalize(description)}"


def fingerprint(value: object) -> str:
    return hashlib.sha1(normalize(value).encode("utf-8")).hexdigest()


def canonical(value: object, *, mask_digits: bool = False) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    if mask_digits:
        value = re.sub(r"\d+(?:[.,]\d+)?", " # ", value)
    value = re.sub(r"[^0-9a-zа-я#]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def stable_shingle_hash(category: str, tokens: list[str]) -> int:
    payload = f"{category}\x1f" + "\x1f".join(tokens)
    return int.from_bytes(
        hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "little"
    )


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def build_mapping(
    frame: pd.DataFrame,
    donors: np.ndarray,
    key: str,
    minimum: int,
    confidence_min: float,
) -> dict[str, int]:
    groups = frame.iloc[donors].groupby(key, sort=False).label.agg(["count", "mean"])
    confidence = np.maximum(groups["mean"], 1 - groups["mean"])
    keep = (
        (groups["count"] >= minimum)
        & (confidence >= confidence_min)
        & (groups["mean"] != 0.5)
        & (groups.index.astype(str) != "")
    )
    selected = groups.loc[keep, "mean"]
    return {str(value): int(mean >= 0.5) for value, mean in selected.items()}


def apply_key_mapping(
    frame: pd.DataFrame,
    targets: np.ndarray,
    predictions: np.ndarray,
    unresolved: np.ndarray,
    mapping: dict[str, int],
    key: str,
) -> int:
    hits = 0
    values = frame.iloc[targets][key].astype(str).to_numpy()
    for local_index, value in enumerate(values):
        if unresolved[local_index] and value in mapping:
            predictions[local_index] = mapping[value]
            unresolved[local_index] = False
            hits += 1
    return hits


def base_prior_predictions(
    frame: pd.DataFrame,
    base: np.ndarray,
    donors: np.ndarray,
    targets: np.ndarray,
    category: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    exact_min, exact_conf, name_min, name_conf = BASE_CONFIGS[category]
    predictions = base[targets].copy()
    unresolved = np.ones(len(targets), dtype=bool)
    hits: dict[str, int] = {}
    exact = build_mapping(frame, donors, "text_hash", exact_min, exact_conf)
    hits["exact"] = apply_key_mapping(
        frame, targets, predictions, unresolved, exact, "text_hash"
    )
    if name_min < 999:
        name = build_mapping(frame, donors, "normalized_name", name_min, name_conf)
        hits["name"] = apply_key_mapping(
            frame, targets, predictions, unresolved, name, "normalized_name"
        )
    else:
        hits["name"] = 0
    canonical_config = CANONICAL_CONFIGS[category]
    if canonical_config is not None:
        key, minimum, confidence = canonical_config
        mapping = build_mapping(frame, donors, key, minimum, confidence)
        hits["canonical"] = apply_key_mapping(
            frame, targets, predictions, unresolved, mapping, key
        )
    else:
        hits["canonical"] = 0
    return predictions, unresolved, hits


def build_neighbor_graph(frame: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    shingle_sets: list[set[int]] = []
    frequencies: Counter[int] = Counter()
    categories = frame.category.astype(str).to_numpy()
    for index, row in enumerate(frame.itertuples(index=False)):
        tokens = canonical(f"{row.name} {row.description}", mask_digits=True).split()
        shingles = {
            stable_shingle_hash(categories[index], tokens[start:start + 5])
            for start in range(max(0, len(tokens) - 4))
        }
        shingle_sets.append(shingles)
        frequencies.update(shingles)
    useful = {key for key, count in frequencies.items() if 2 <= count <= 30}
    postings: dict[int, list[int]] = defaultdict(list)
    for index, shingles in enumerate(shingle_sets):
        for key in shingles & useful:
            postings[key].append(index)
    neighbor_indices: list[np.ndarray] = []
    neighbor_scores: list[np.ndarray] = []
    for index, shingles in enumerate(shingle_sets):
        counts: Counter[int] = Counter()
        useful_shingles = shingles & useful
        for key in useful_shingles:
            for candidate in postings[key]:
                if candidate != index:
                    counts[candidate] += 1
        rows = []
        for candidate, intersection in counts.items():
            if categories[candidate] != categories[index] or intersection < 8:
                continue
            candidate_size = len(shingle_sets[candidate])
            minimum_size = min(len(shingles), candidate_size)
            union_size = len(shingles) + candidate_size - intersection
            if not minimum_size or not union_size:
                continue
            containment = intersection / minimum_size
            jaccard = intersection / union_size
            if containment >= 0.40 and jaccard >= 0.20:
                rows.append((containment, jaccard, intersection, candidate))
        rows.sort(reverse=True)
        rows = rows[:50]
        neighbor_indices.append(np.asarray([row[3] for row in rows], dtype=np.int32))
        neighbor_scores.append(np.asarray([row[:3] for row in rows], dtype=np.float32))
        if (index + 1) % 1000 == 0 or index + 1 == len(frame):
            print(
                f"neighbor_graph={index + 1}/{len(frame)} useful_shingles={len(useful):,}",
                flush=True,
            )
    return neighbor_indices, neighbor_scores


def apply_neighbor_prior(
    frame: pd.DataFrame,
    donors: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    unresolved: np.ndarray,
    neighbor_indices: list[np.ndarray],
    neighbor_scores: list[np.ndarray],
    config: tuple[float, int, int, float],
) -> tuple[np.ndarray, int, int]:
    threshold, minimum_neighbors, top_k, confidence_min = config
    donor_mask = np.zeros(len(frame), dtype=bool)
    donor_mask[donors] = True
    labels = frame.label.to_numpy(dtype=np.int8)
    changed = 0
    hits = 0
    result = predictions.copy()
    for local_index, target in enumerate(targets):
        if not unresolved[local_index]:
            continue
        indices = neighbor_indices[target]
        scores = neighbor_scores[target]
        selected = []
        for neighbor, score in zip(indices, scores):
            if score[0] < threshold:
                continue
            if donor_mask[neighbor]:
                selected.append(int(neighbor))
                if len(selected) >= top_k:
                    break
        if len(selected) < minimum_neighbors:
            continue
        mean = float(labels[selected].mean())
        confidence = max(mean, 1 - mean)
        if confidence < confidence_min or mean == 0.5:
            continue
        prediction = int(mean >= 0.5)
        hits += 1
        if prediction != result[local_index]:
            changed += 1
        result[local_index] = prediction
    return result, hits, changed


def config_grid(category: str) -> list[tuple[float, int, int, float]]:
    thresholds = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    minimums = [1, 2, 3]
    top_ks = [1, 3, 5]
    confidences = [0.75, 0.999]
    configs = []
    for threshold in thresholds:
        for minimum in minimums:
            for top_k in top_ks:
                if minimum > top_k:
                    continue
                for confidence in confidences:
                    if category == "Легковоспламеняющиеся" and confidence < 0.999:
                        continue
                    configs.append((threshold, minimum, top_k, confidence))
    return configs


def choose_nested(
    frame: pd.DataFrame,
    base: np.ndarray,
    category: str,
    outer_fold: int,
    neighbor_indices: list[np.ndarray],
    neighbor_scores: list[np.ndarray],
) -> tuple[tuple[float, int, int, float], float, int, int]:
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)
    category_mask = frame.category.to_numpy() == category
    outer_train = category_mask & (folds != outer_fold)
    best = None
    for config in config_grid(category):
        inner_labels, inner_predictions = [], []
        total_hits = total_changed = 0
        for inner_fold in sorted(frame.fold.unique()):
            if inner_fold == outer_fold:
                continue
            donors = np.flatnonzero(outer_train & (folds != inner_fold))
            targets = np.flatnonzero(outer_train & (folds == inner_fold))
            baseline, unresolved, _ = base_prior_predictions(
                frame, base, donors, targets, category
            )
            predictions, hits, changed = apply_neighbor_prior(
                frame, donors, targets, baseline, unresolved,
                neighbor_indices, neighbor_scores, config,
            )
            inner_labels.append(labels[targets])
            inner_predictions.append(predictions)
            total_hits += hits
            total_changed += changed
        value = f1(np.concatenate(inner_labels), np.concatenate(inner_predictions))
        preference = (
            value, -total_changed, -total_hits,
            config[0], config[1], -config[2], config[3],
        )
        if best is None or preference > best[0]:
            best = (preference, config, value, total_hits, total_changed)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def evaluate_oof(
    frame: pd.DataFrame,
    base: np.ndarray,
    category: str,
    neighbor_indices: list[np.ndarray],
    neighbor_scores: list[np.ndarray],
    config: tuple[float, int, int, float] | None,
) -> tuple[float, int, int, np.ndarray]:
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)
    category_mask = frame.category.to_numpy() == category
    all_predictions = base.copy()
    total_hits = total_changed = 0
    for fold in sorted(frame.fold.unique()):
        donors = np.flatnonzero(category_mask & (folds != fold))
        targets = np.flatnonzero(category_mask & (folds == fold))
        baseline, unresolved, _ = base_prior_predictions(frame, base, donors, targets, category)
        if config is None:
            predictions = baseline
        else:
            predictions, hits, changed = apply_neighbor_prior(
                frame, donors, targets, baseline, unresolved,
                neighbor_indices, neighbor_scores, config,
            )
            total_hits += hits
            total_changed += changed
        all_predictions[targets] = predictions
    positions = np.flatnonzero(category_mask)
    return f1(labels[positions], all_predictions[positions]), total_hits, total_changed, all_predictions


def repeated_random_split(
    frame: pd.DataFrame,
    base: np.ndarray,
    category: str,
    neighbor_indices: list[np.ndarray],
    neighbor_scores: list[np.ndarray],
    config: tuple[float, int, int, float],
) -> dict[str, object]:
    labels = frame.label.to_numpy(dtype=np.int8)
    category_positions = np.flatnonzero(frame.category.to_numpy() == category)
    rng = np.random.default_rng(20260821)
    rows = []
    for repeat in range(20):
        train_parts, valid_parts = [], []
        for label in [0, 1]:
            positions = category_positions[labels[category_positions] == label].copy()
            rng.shuffle(positions)
            cut = int(round(0.70 * len(positions)))
            train_parts.append(positions[:cut])
            valid_parts.append(positions[cut:])
        donors = np.concatenate(train_parts)
        targets = np.concatenate(valid_parts)
        baseline, unresolved, _ = base_prior_predictions(frame, base, donors, targets, category)
        candidate, hits, changed = apply_neighbor_prior(
            frame, donors, targets, baseline, unresolved,
            neighbor_indices, neighbor_scores, config,
        )
        rows.append({
            "repeat": repeat,
            "rows": int(len(targets)),
            "baseline_f1": f1(labels[targets], baseline),
            "candidate_f1": f1(labels[targets], candidate),
            "hits": hits,
            "changed": changed,
        })
    deltas = np.asarray([row["candidate_f1"] - row["baseline_f1"] for row in rows])
    return {
        "baseline_mean_f1": float(np.mean([row["baseline_f1"] for row in rows])),
        "candidate_mean_f1": float(np.mean([row["candidate_f1"] for row in rows])),
        "mean_delta": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "positive_repeats": int((deltas > 0).sum()),
        "negative_repeats": int((deltas < 0).sum()),
        "mean_hits": float(np.mean([row["hits"] for row in rows])),
        "mean_changed": float(np.mean([row["changed"] for row in rows])),
        "repeats": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", default=str(FUSION))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()
    frame = pd.read_csv(DATA)
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["normalized_name"] = frame.name.map(normalize)
    texts = [compose_text(name, description) for name, description in zip(frame.name, frame.description)]
    frame["text_hash"] = [fingerprint(text) for text in texts]
    frame["canonical_text_mask_digits"] = [
        canonical(text, mask_digits=True) for text in texts
    ]
    fusion = np.load(args.fusion, allow_pickle=True)
    if not np.array_equal(frame.id.astype(str).to_numpy(), fusion["ids"].astype(str)):
        raise ValueError("id mismatch")
    frame["fold"] = fusion["folds"].astype(np.int8)
    base = fusion["full_oof_predictions"].astype(np.int8)
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)
    neighbor_indices, neighbor_scores = build_neighbor_graph(frame)
    print(
        f"graph_rows_with_neighbors={sum(bool(x.size) for x in neighbor_indices)} "
        f"graph_edges={sum(len(x) for x in neighbor_indices)}",
        flush=True,
    )
    report: dict[str, object] = {
        "protocol": "label-free rare 5-word-shingle graph; donor-only nested grouped CV after exact/name/canonical prior",
        "graph": {
            "rows_with_neighbors": sum(bool(x.size) for x in neighbor_indices),
            "directed_edges": sum(len(x) for x in neighbor_indices),
        },
        "categories": {},
    }
    macro_baseline, macro_nested, macro_fixed = [], [], []
    audit_rows: list[dict[str, object]] = []
    for category in sorted(frame.category.unique()):
        category_mask = frame.category.to_numpy() == category
        category_positions = np.flatnonzero(category_mask)
        baseline_f1, _, _, baseline_predictions = evaluate_oof(
            frame, base, category, neighbor_indices, neighbor_scores, None
        )
        nested_predictions = base.copy()
        choices = []
        fold_rows = []
        for outer_fold in sorted(frame.fold.unique()):
            config, inner_f1, inner_hits, inner_changed = choose_nested(
                frame, base, category, int(outer_fold), neighbor_indices, neighbor_scores
            )
            donors = np.flatnonzero(category_mask & (folds != outer_fold))
            targets = np.flatnonzero(category_mask & (folds == outer_fold))
            baseline, unresolved, _ = base_prior_predictions(frame, base, donors, targets, category)
            predictions, hits, changed = apply_neighbor_prior(
                frame, donors, targets, baseline, unresolved,
                neighbor_indices, neighbor_scores, config,
            )
            nested_predictions[targets] = predictions
            choices.append(config)
            fold_rows.append({
                "fold": int(outer_fold),
                "selected": list(config),
                "inner_f1": inner_f1,
                "inner_hits": inner_hits,
                "inner_changed": inner_changed,
                "validation_f1": f1(labels[targets], predictions),
                "validation_hits": hits,
                "validation_changed": changed,
            })
        nested_f1 = f1(labels[category_positions], nested_predictions[category_positions])
        choice_counts = Counter(choices)
        selected = max(
            choice_counts,
            key=lambda row: (choice_counts[row], row[0], row[1], -row[2], row[3]),
        )
        fixed_f1, fixed_hits, fixed_changed, fixed_predictions = evaluate_oof(
            frame, base, category, neighbor_indices, neighbor_scores, selected
        )
        random_split = repeated_random_split(
            frame, base, category, neighbor_indices, neighbor_scores, selected
        )
        safety_rows = []
        safety_configs = [
            selected,
            (0.95, 2, 3, 0.999),
            (0.95, 3, 3, 0.999),
            (0.85, 3, 3, 0.999),
        ]
        for safety_config in dict.fromkeys(safety_configs):
            safety_f1, safety_hits, safety_changed, safety_predictions = evaluate_oof(
                frame, base, category, neighbor_indices, neighbor_scores, safety_config
            )
            safety_changed_mask = (
                safety_predictions[category_positions]
                != baseline_predictions[category_positions]
            )
            safety_positions = category_positions[safety_changed_mask]
            safety_random = repeated_random_split(
                frame, base, category, neighbor_indices, neighbor_scores, safety_config
            )
            safety_rows.append({
                "config": list(safety_config),
                "f1": safety_f1,
                "delta": safety_f1 - baseline_f1,
                "hits": safety_hits,
                "changed": safety_changed,
                "changed_before_correct": int(
                    (baseline_predictions[safety_positions] == labels[safety_positions]).sum()
                ),
                "changed_after_correct": int(
                    (safety_predictions[safety_positions] == labels[safety_positions]).sum()
                ),
                "random_split": {
                    key: safety_random[key]
                    for key in [
                        "baseline_mean_f1", "candidate_mean_f1", "mean_delta",
                        "delta_std", "positive_repeats", "negative_repeats",
                        "mean_hits", "mean_changed",
                    ]
                },
            })
        changed_mask = (
            fixed_predictions[category_positions] != baseline_predictions[category_positions]
        )
        changed_positions = category_positions[changed_mask]
        if fixed_changed:
            threshold, _, top_k, _ = selected
            for target in changed_positions:
                donor_mask = category_mask & (folds != folds[target])
                neighbors = []
                for neighbor, score in zip(neighbor_indices[target], neighbor_scores[target]):
                    if score[0] < threshold:
                        continue
                    if donor_mask[neighbor]:
                        neighbors.append({
                            "id": str(frame.id.iloc[neighbor]),
                            "label": int(labels[neighbor]),
                            "containment": float(score[0]),
                            "jaccard": float(score[1]),
                            "name": str(frame.name.iloc[neighbor])[:240],
                        })
                        if len(neighbors) >= top_k:
                            break
                audit_rows.append({
                    "category": category,
                    "id": str(frame.id.iloc[target]),
                    "fold": int(folds[target]),
                    "label": int(labels[target]),
                    "before": int(baseline_predictions[target]),
                    "after": int(fixed_predictions[target]),
                    "before_correct": bool(baseline_predictions[target] == labels[target]),
                    "after_correct": bool(fixed_predictions[target] == labels[target]),
                    "name": str(frame.name.iloc[target]),
                    "description": str(frame.description.iloc[target])[:1200],
                    "neighbors": json.dumps(neighbors, ensure_ascii=False),
                })
        report["categories"][category] = {
            "baseline_f1": baseline_f1,
            "nested_f1": nested_f1,
            "nested_delta": nested_f1 - baseline_f1,
            "selected_full_config": list(selected),
            "choice_counts": {str(list(key)): value for key, value in choice_counts.items()},
            "fixed_f1": fixed_f1,
            "fixed_delta": fixed_f1 - baseline_f1,
            "fixed_hits": fixed_hits,
            "fixed_changed": fixed_changed,
            "changed_before_correct": int(
                (baseline_predictions[changed_positions] == labels[changed_positions]).sum()
            ),
            "changed_after_correct": int(
                (fixed_predictions[changed_positions] == labels[changed_positions]).sum()
            ),
            "changed_ids": frame.id.iloc[changed_positions].astype(str).tolist(),
            "safety_configs": safety_rows,
            "folds": fold_rows,
            "random_split": random_split,
        }
        macro_baseline.append(baseline_f1)
        macro_nested.append(nested_f1)
        macro_fixed.append(fixed_f1)
    report["macro"] = {
        "baseline_f1": float(np.mean(macro_baseline)),
        "nested_f1": float(np.mean(macro_nested)),
        "nested_delta": float(np.mean(macro_nested) - np.mean(macro_baseline)),
        "fixed_f1": float(np.mean(macro_fixed)),
        "fixed_delta": float(np.mean(macro_fixed) - np.mean(macro_baseline)),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(AUDIT, sep="\t", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
