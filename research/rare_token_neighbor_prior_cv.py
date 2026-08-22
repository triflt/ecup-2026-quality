from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from char_tfidf_neighbor_prior_cv import (
    apply_char_prior,
    choose_nested,
    config_grid,
    current_prior_predictions,
    evaluate_oof,
    repeated_random_split,
)
from shingle_neighbor_prior_cv import (
    DATA,
    FUSION,
    build_neighbor_graph,
    canonical,
    compose_text,
    f1,
    fingerprint,
    normalize,
)


ROOT = Path("research")
REPORT = ROOT / "qwen3vl-qwen35-rare-token-prior-report.json"
AUDIT = ROOT / "qwen3vl-qwen35-rare-token-audit.tsv"


def feature_hash(category: str, kind: str, value: str) -> int:
    payload = f"{category}\x1f{kind}\x1f{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def row_features(category: str, name: object, description: object) -> set[int]:
    name_tokens = canonical(name, mask_digits=False).split()
    description_tokens = canonical(description, mask_digits=False).split()
    tokens = name_tokens * 2 + description_tokens
    unigram = {
        feature_hash(category, "u", token)
        for token in tokens
        if len(token) >= 3 or any(character.isdigit() for character in token)
    }
    bigram = {
        feature_hash(category, "b", f"{left}\x1f{right}")
        for left, right in zip(tokens, tokens[1:])
        if len(left) + len(right) >= 7
    }
    return unigram | bigram


def build_rare_token_graph(
    frame: pd.DataFrame,
    *,
    maximum_frequency: int = 40,
    neighbors: int = 50,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, object]]:
    features: list[set[int]] = []
    frequencies: Counter[int] = Counter()
    categories = frame.category.astype(str).to_numpy()
    for index, row in enumerate(frame.itertuples(index=False)):
        values = row_features(categories[index], row.name, row.description)
        features.append(values)
        frequencies.update(values)
    useful = {
        value
        for value, count in frequencies.items()
        if 2 <= count <= maximum_frequency
    }
    weights = {
        value: math.log((len(frame) + 1) / (frequencies[value] + 1)) + 1.0
        for value in useful
    }
    postings: dict[int, list[int]] = defaultdict(list)
    totals = np.zeros(len(frame), dtype=np.float32)
    useful_counts = np.zeros(len(frame), dtype=np.int32)
    for index, values in enumerate(features):
        selected = values & useful
        useful_counts[index] = len(selected)
        totals[index] = sum(weights[value] for value in selected)
        for value in selected:
            postings[value].append(index)

    graph_indices: list[np.ndarray] = []
    graph_scores: list[np.ndarray] = []
    directed_edges = rows_with_neighbors = 0
    for index, values in enumerate(features):
        intersections: Counter[int] = Counter()
        shared_counts: Counter[int] = Counter()
        for value in values & useful:
            weight = weights[value]
            for candidate in postings[value]:
                if candidate == index:
                    continue
                intersections[candidate] += weight
                shared_counts[candidate] += 1
        rows = []
        for candidate, intersection in intersections.items():
            if categories[candidate] != categories[index]:
                continue
            if shared_counts[candidate] < 4:
                continue
            minimum = min(float(totals[index]), float(totals[candidate]))
            union = float(totals[index] + totals[candidate] - intersection)
            if minimum <= 0 or union <= 0:
                continue
            containment = float(intersection) / minimum
            jaccard = float(intersection) / union
            score = min(containment, 2.0 * jaccard, 1.0)
            if score < 0.70:
                continue
            rows.append(
                (
                    score,
                    containment,
                    jaccard,
                    int(shared_counts[candidate]),
                    candidate,
                )
            )
        rows.sort(reverse=True)
        rows = rows[:neighbors]
        graph_indices.append(
            np.asarray([row[4] for row in rows], dtype=np.int32)
        )
        graph_scores.append(
            np.asarray([row[0] for row in rows], dtype=np.float32)
        )
        directed_edges += len(rows)
        rows_with_neighbors += bool(rows)
        if (index + 1) % 1000 == 0 or index + 1 == len(frame):
            print(
                f"rare_token_graph={index + 1}/{len(frame)} "
                f"useful_features={len(useful):,} edges={directed_edges:,}",
                flush=True,
            )
    details = {
        "useful_features": int(len(useful)),
        "posting_entries": int(sum(len(rows) for rows in postings.values())),
        "rows_with_neighbors": int(rows_with_neighbors),
        "directed_edges": int(directed_edges),
        "mean_useful_features": float(useful_counts.mean()),
        "maximum_feature_frequency": maximum_frequency,
        "minimum_shared_features": 4,
        "score": "min(weighted_containment, 2 * weighted_jaccard)",
    }
    return graph_indices, graph_scores, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", default=str(FUSION))
    parser.add_argument("--report", default=str(REPORT))
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
    token_indices, token_scores, graph_details = build_rare_token_graph(frame)
    report: dict[str, object] = {
        "protocol": (
            "label-free order-independent rare unigram/bigram graph; donor-only "
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
            token_indices,
            token_scores,
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
                token_indices,
                token_scores,
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
                token_indices,
                token_scores,
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
        fixed_predictions: dict[tuple[float, int, int, float], np.ndarray] = {}
        for config in config_grid():
            if config[1] < 2:
                continue
            value, hits, changed, predictions = evaluate_oof(
                frame,
                base,
                category,
                shingle_indices,
                shingle_scores,
                token_indices,
                token_scores,
                config,
            )
            fixed_predictions[config] = predictions
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
                token_indices,
                token_scores,
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
            predictions = fixed_predictions[production_config]
            changed_positions = category_positions[
                predictions[category_positions]
                != baseline_predictions[category_positions]
            ]
            threshold, _, top_k, _ = production_config
            for target in changed_positions:
                donor_mask = category_mask & (folds != folds[target])
                neighbors = []
                for neighbor, score in zip(
                    token_indices[target], token_scores[target]
                ):
                    if float(score) < threshold:
                        break
                    if donor_mask[neighbor]:
                        neighbors.append(
                            {
                                "id": str(frame.id.iloc[neighbor]),
                                "label": int(labels[neighbor]),
                                "score": float(score),
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
                        "after": int(predictions[target]),
                        "before_correct": bool(
                            baseline_predictions[target] == labels[target]
                        ),
                        "after_correct": bool(predictions[target] == labels[target]),
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
