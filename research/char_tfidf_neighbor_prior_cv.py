from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from shingle_neighbor_prior_cv import (
    DATA,
    FUSION,
    base_prior_predictions,
    build_neighbor_graph,
    canonical,
    compose_text,
    f1,
    fingerprint,
    normalize,
)


ROOT = Path("research")
DATA = Path(os.environ.get("ECUP_DATA", ROOT / "data.csv"))
REPORT = ROOT / "qwen3vl-qwen35-char-neighbor-prior-report.json"
AUDIT = ROOT / "qwen3vl-qwen35-char-neighbor-audit.tsv"

SHINGLE_CONFIGS: dict[str, tuple[float, int, int, float] | None] = {
    "БАД": (0.95, 2, 3, 0.999),
    "Легковоспламеняющиеся": None,
}


def build_char_graph(
    frame: pd.DataFrame,
    *,
    max_features: int = 250_000,
    neighbors: int = 50,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, object]]:
    graph_indices = [np.empty(0, dtype=np.int32) for _ in range(len(frame))]
    graph_scores = [np.empty(0, dtype=np.float32) for _ in range(len(frame))]
    details: dict[str, object] = {}
    categories = frame.category.astype(str).to_numpy()
    for category in sorted(frame.category.unique()):
        positions = np.flatnonzero(categories == category)
        texts = []
        for position in positions:
            row = frame.iloc[position]
            name = canonical(row["name"], mask_digits=True)
            description = canonical(row["description"], mask_digits=True)
            texts.append(f"{name} {name} {name} {description}")
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.995,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(texts)
        count = min(neighbors + 1, len(positions))
        search = NearestNeighbors(
            n_neighbors=count,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(matrix)
        edge_count = 0
        for start in range(0, len(positions), 256):
            stop = min(start + 256, len(positions))
            distances, local_indices = search.kneighbors(matrix[start:stop])
            for offset, (distance_row, local_row) in enumerate(
                zip(distances, local_indices)
            ):
                local_target = start + offset
                keep = local_row != local_target
                local_row = local_row[keep][:neighbors]
                similarities = (1.0 - distance_row[keep][:neighbors]).astype(
                    np.float32
                )
                global_target = int(positions[local_target])
                graph_indices[global_target] = positions[local_row].astype(np.int32)
                graph_scores[global_target] = similarities
                edge_count += len(local_row)
            if stop % 1024 == 0 or stop == len(positions):
                print(
                    f"char_graph category={category} rows={stop}/{len(positions)} "
                    f"features={matrix.shape[1]:,} nnz={matrix.nnz:,}",
                    flush=True,
                )
        details[str(category)] = {
            "rows": int(len(positions)),
            "features": int(matrix.shape[1]),
            "nnz": int(matrix.nnz),
            "directed_edges": int(edge_count),
        }
    return graph_indices, graph_scores, details


def apply_shingle_and_mark(
    frame: pd.DataFrame,
    donors: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    unresolved: np.ndarray,
    neighbor_indices: list[np.ndarray],
    neighbor_scores: list[np.ndarray],
    config: tuple[float, int, int, float] | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if config is None:
        return predictions.copy(), unresolved.copy(), 0
    threshold, minimum_neighbors, top_k, confidence_min = config
    donor_mask = np.zeros(len(frame), dtype=bool)
    donor_mask[donors] = True
    labels = frame.label.to_numpy(dtype=np.int8)
    result = predictions.copy()
    remaining = unresolved.copy()
    hits = 0
    for local_index, target in enumerate(targets):
        if not remaining[local_index]:
            continue
        selected: list[int] = []
        for neighbor, score in zip(
            neighbor_indices[target], neighbor_scores[target]
        ):
            if float(score[0]) < threshold:
                continue
            if donor_mask[neighbor]:
                selected.append(int(neighbor))
                if len(selected) >= top_k:
                    break
        if len(selected) < minimum_neighbors:
            continue
        mean = float(labels[selected].mean())
        confidence = max(mean, 1.0 - mean)
        if confidence < confidence_min or mean == 0.5:
            continue
        result[local_index] = int(mean >= 0.5)
        remaining[local_index] = False
        hits += 1
    return result, remaining, hits


def current_prior_predictions(
    frame: pd.DataFrame,
    base: np.ndarray,
    donors: np.ndarray,
    targets: np.ndarray,
    category: str,
    shingle_indices: list[np.ndarray],
    shingle_scores: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    predictions, unresolved, hits = base_prior_predictions(
        frame, base, donors, targets, category
    )
    predictions, unresolved, shingle_hits = apply_shingle_and_mark(
        frame,
        donors,
        targets,
        predictions,
        unresolved,
        shingle_indices,
        shingle_scores,
        SHINGLE_CONFIGS[category],
    )
    hits["shingle"] = shingle_hits
    return predictions, unresolved, hits


def apply_char_prior(
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
    result = predictions.copy()
    hits = changed = 0
    for local_index, target in enumerate(targets):
        if not unresolved[local_index]:
            continue
        selected: list[int] = []
        for neighbor, score in zip(
            neighbor_indices[target], neighbor_scores[target]
        ):
            if float(score) < threshold:
                break
            if donor_mask[neighbor]:
                selected.append(int(neighbor))
                if len(selected) >= top_k:
                    break
        if len(selected) < minimum_neighbors:
            continue
        mean = float(labels[selected].mean())
        confidence = max(mean, 1.0 - mean)
        if confidence < confidence_min or mean == 0.5:
            continue
        prediction = int(mean >= 0.5)
        hits += 1
        if prediction != result[local_index]:
            changed += 1
        result[local_index] = prediction
    return result, hits, changed


def config_grid() -> list[tuple[float, int, int, float]]:
    configs = []
    for threshold in [0.90, 0.93, 0.95, 0.97, 0.98, 0.99]:
        for minimum, top_k in [(1, 1), (2, 3), (2, 5), (3, 3), (3, 5)]:
            configs.append((threshold, minimum, top_k, 0.999))
    return configs


def evaluate_oof(
    frame: pd.DataFrame,
    base: np.ndarray,
    category: str,
    shingle_indices: list[np.ndarray],
    shingle_scores: list[np.ndarray],
    char_indices: list[np.ndarray],
    char_scores: list[np.ndarray],
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
        baseline, unresolved, _ = current_prior_predictions(
            frame,
            base,
            donors,
            targets,
            category,
            shingle_indices,
            shingle_scores,
        )
        if config is None:
            predictions = baseline
        else:
            predictions, hits, changed = apply_char_prior(
                frame,
                donors,
                targets,
                baseline,
                unresolved,
                char_indices,
                char_scores,
                config,
            )
            total_hits += hits
            total_changed += changed
        all_predictions[targets] = predictions
    positions = np.flatnonzero(category_mask)
    return (
        f1(labels[positions], all_predictions[positions]),
        total_hits,
        total_changed,
        all_predictions,
    )


def choose_nested(
    frame: pd.DataFrame,
    base: np.ndarray,
    category: str,
    outer_fold: int,
    shingle_indices: list[np.ndarray],
    shingle_scores: list[np.ndarray],
    char_indices: list[np.ndarray],
    char_scores: list[np.ndarray],
) -> tuple[tuple[float, int, int, float], float, int, int]:
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)
    category_mask = frame.category.to_numpy() == category
    outer_train = category_mask & (folds != outer_fold)
    best = None
    for config in config_grid():
        inner_labels, inner_predictions = [], []
        total_hits = total_changed = 0
        for inner_fold in sorted(frame.fold.unique()):
            if inner_fold == outer_fold:
                continue
            donors = np.flatnonzero(outer_train & (folds != inner_fold))
            targets = np.flatnonzero(outer_train & (folds == inner_fold))
            baseline, unresolved, _ = current_prior_predictions(
                frame,
                base,
                donors,
                targets,
                category,
                shingle_indices,
                shingle_scores,
            )
            predictions, hits, changed = apply_char_prior(
                frame,
                donors,
                targets,
                baseline,
                unresolved,
                char_indices,
                char_scores,
                config,
            )
            inner_labels.append(labels[targets])
            inner_predictions.append(predictions)
            total_hits += hits
            total_changed += changed
        value = f1(np.concatenate(inner_labels), np.concatenate(inner_predictions))
        preference = (
            value,
            -total_changed,
            -total_hits,
            config[0],
            config[1],
            -config[2],
        )
        if best is None or preference > best[0]:
            best = (preference, config, value, total_hits, total_changed)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def repeated_random_split(
    frame: pd.DataFrame,
    base: np.ndarray,
    category: str,
    shingle_indices: list[np.ndarray],
    shingle_scores: list[np.ndarray],
    char_indices: list[np.ndarray],
    char_scores: list[np.ndarray],
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
        baseline, unresolved, _ = current_prior_predictions(
            frame,
            base,
            donors,
            targets,
            category,
            shingle_indices,
            shingle_scores,
        )
        candidate, hits, changed = apply_char_prior(
            frame,
            donors,
            targets,
            baseline,
            unresolved,
            char_indices,
            char_scores,
            config,
        )
        rows.append(
            {
                "repeat": repeat,
                "rows": int(len(targets)),
                "baseline_f1": f1(labels[targets], baseline),
                "candidate_f1": f1(labels[targets], candidate),
                "hits": hits,
                "changed": changed,
            }
        )
    deltas = np.asarray(
        [row["candidate_f1"] - row["baseline_f1"] for row in rows]
    )
    return {
        "baseline_mean_f1": float(
            np.mean([row["baseline_f1"] for row in rows])
        ),
        "candidate_mean_f1": float(
            np.mean([row["candidate_f1"] for row in rows])
        ),
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
    parser.add_argument("--max-features", type=int, default=250_000)
    args = parser.parse_args()

    frame = pd.read_csv(DATA)
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["normalized_name"] = frame.name.map(normalize)
    texts = [
        compose_text(name, description)
        for name, description in zip(frame.name, frame.description)
    ]
    frame["text_hash"] = [fingerprint(text) for text in texts]
    frame["canonical_text_mask_digits"] = [
        canonical(text, mask_digits=True) for text in texts
    ]

    fusion = np.load(args.fusion, allow_pickle=True)
    if not np.array_equal(
        frame.id.astype(str).to_numpy(), fusion["ids"].astype(str)
    ):
        raise ValueError("id mismatch")
    frame["fold"] = fusion["folds"].astype(np.int8)
    base = fusion["full_oof_predictions"].astype(np.int8)
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)

    shingle_indices, shingle_scores = build_neighbor_graph(frame)
    char_indices, char_scores, graph_details = build_char_graph(
        frame, max_features=args.max_features
    )
    report: dict[str, object] = {
        "protocol": (
            "label-free masked-digit character TF-IDF neighbours; donor-only "
            "nested grouped CV after exact/name/canonical and production shingle prior"
        ),
        "graph": graph_details,
        "categories": {},
    }
    audit_rows: list[dict[str, object]] = []
    macro_baseline, macro_nested = [], []

    for category in sorted(frame.category.unique()):
        category_mask = frame.category.to_numpy() == category
        category_positions = np.flatnonzero(category_mask)
        baseline_f1, _, _, baseline_predictions = evaluate_oof(
            frame,
            base,
            category,
            shingle_indices,
            shingle_scores,
            char_indices,
            char_scores,
            None,
        )
        nested_predictions = base.copy()
        choices = []
        fold_rows = []
        for outer_fold in sorted(frame.fold.unique()):
            config, inner_f1, inner_hits, inner_changed = choose_nested(
                frame,
                base,
                category,
                int(outer_fold),
                shingle_indices,
                shingle_scores,
                char_indices,
                char_scores,
            )
            donors = np.flatnonzero(category_mask & (folds != outer_fold))
            targets = np.flatnonzero(category_mask & (folds == outer_fold))
            baseline, unresolved, _ = current_prior_predictions(
                frame,
                base,
                donors,
                targets,
                category,
                shingle_indices,
                shingle_scores,
            )
            predictions, hits, changed = apply_char_prior(
                frame,
                donors,
                targets,
                baseline,
                unresolved,
                char_indices,
                char_scores,
                config,
            )
            nested_predictions[targets] = predictions
            choices.append(config)
            fold_rows.append(
                {
                    "fold": int(outer_fold),
                    "selected": list(config),
                    "inner_f1": inner_f1,
                    "inner_hits": inner_hits,
                    "inner_changed": inner_changed,
                    "validation_f1": f1(labels[targets], predictions),
                    "validation_hits": hits,
                    "validation_changed": changed,
                }
            )
        nested_f1 = f1(
            labels[category_positions], nested_predictions[category_positions]
        )
        choice_counts = Counter(choices)

        fixed_rows = []
        for config in config_grid():
            if config[1] < 2:
                continue
            value, hits, changed, predictions = evaluate_oof(
                frame,
                base,
                category,
                shingle_indices,
                shingle_scores,
                char_indices,
                char_scores,
                config,
            )
            changed_positions = category_positions[
                predictions[category_positions]
                != baseline_predictions[category_positions]
            ]
            random_split = repeated_random_split(
                frame,
                base,
                category,
                shingle_indices,
                shingle_scores,
                char_indices,
                char_scores,
                config,
            )
            fixed_rows.append(
                {
                    "config": list(config),
                    "f1": value,
                    "delta": value - baseline_f1,
                    "hits": hits,
                    "changed": changed,
                    "changed_before_correct": int(
                        (
                            baseline_predictions[changed_positions]
                            == labels[changed_positions]
                        ).sum()
                    ),
                    "changed_after_correct": int(
                        (
                            predictions[changed_positions]
                            == labels[changed_positions]
                        ).sum()
                    ),
                    "random_split": random_split,
                }
            )

        stable = [
            row
            for row in fixed_rows
            if row["delta"] > 0
            and row["changed_after_correct"] > row["changed_before_correct"]
            and row["random_split"]["positive_repeats"] >= 18
            and row["random_split"]["negative_repeats"] <= 1
        ]
        production = max(
            stable,
            key=lambda row: (
                row["random_split"]["mean_delta"],
                row["delta"],
                -row["changed"],
            ),
            default=None,
        )
        if production is not None:
            production_config = tuple(production["config"])
            _, _, _, production_predictions = evaluate_oof(
                frame,
                base,
                category,
                shingle_indices,
                shingle_scores,
                char_indices,
                char_scores,
                production_config,
            )
            changed_positions = category_positions[
                production_predictions[category_positions]
                != baseline_predictions[category_positions]
            ]
            threshold, _, top_k, _ = production_config
            for target in changed_positions:
                donor_mask = category_mask & (folds != folds[target])
                neighbors = []
                for neighbor, score in zip(
                    char_indices[target], char_scores[target]
                ):
                    if float(score) < threshold:
                        break
                    if donor_mask[neighbor]:
                        neighbors.append(
                            {
                                "id": str(frame.id.iloc[neighbor]),
                                "label": int(labels[neighbor]),
                                "similarity": float(score),
                                "name": str(frame.name.iloc[neighbor])[:240],
                            }
                        )
                        if len(neighbors) >= top_k:
                            break
                audit_rows.append(
                    {
                        "category": category,
                        "id": str(frame.id.iloc[target]),
                        "fold": int(folds[target]),
                        "label": int(labels[target]),
                        "before": int(baseline_predictions[target]),
                        "after": int(production_predictions[target]),
                        "before_correct": bool(
                            baseline_predictions[target] == labels[target]
                        ),
                        "after_correct": bool(
                            production_predictions[target] == labels[target]
                        ),
                        "name": str(frame.name.iloc[target]),
                        "description": str(frame.description.iloc[target])[:1200],
                        "neighbors": json.dumps(neighbors, ensure_ascii=False),
                    }
                )

        report["categories"][category] = {
            "baseline_f1": baseline_f1,
            "nested_f1": nested_f1,
            "nested_delta": nested_f1 - baseline_f1,
            "choice_counts": {
                str(list(key)): value for key, value in choice_counts.items()
            },
            "folds": fold_rows,
            "production": production,
            "fixed_configs": fixed_rows,
        }
        macro_baseline.append(baseline_f1)
        macro_nested.append(nested_f1)

    report["macro"] = {
        "baseline_f1": float(np.mean(macro_baseline)),
        "nested_f1": float(np.mean(macro_nested)),
        "nested_delta": float(np.mean(macro_nested) - np.mean(macro_baseline)),
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(audit_rows).to_csv(AUDIT, sep="\t", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
