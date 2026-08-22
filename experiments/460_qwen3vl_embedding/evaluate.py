from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from qwen35_locked_190_audit import evaluate_locked
from qwen35_seed_ensemble_cv import (
    BASE,
    QWEN3VL,
    QWEN35_SEED_A,
    best_threshold,
    f1,
    fold_category_ranks,
    load_seed_predictions,
    rank01,
)

DATA = ROOT / "research/data.csv"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
ROUTE400 = ROOT / "experiments/400_qwen35_category_routed_adapters/results/routed_predictions.npz"
PARENT_QWEN35 = ROOT / "experiments/260_bad_family_diverse_positives/artifacts/seed_42_diverse_positives"
OUT = Path(__file__).resolve().parent / "results"
MANIFEST = Path(__file__).resolve().parent / "analysis/selection_manifest.json"
BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
WEIGHTS = (0.05, 0.10, 0.15, 0.20)
SAFETY_PATTERN = re.compile(
    r"(?iu)\b(?:зажигалк|спич|огнив|факел|свеч|горелк|фейерверк|салют|"
    r"петард|бенгал|пиротех|дымогенератор|свеч\w*\s*фонтан|"
    r"фонтан\w*\s+для\s+торт|угол|уголь|дров|брик|розжиг)\w*\b"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


connected = load_module("connected_guard_audit_460", ROOT / "research/audit_connected_guard_existing.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crossfit_embedding_scores(
    features: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    positions = np.flatnonzero(categories == FLAMMABLE)
    scores = np.full(len(labels), np.nan, dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        train = positions[folds[positions] != fold]
        valid = positions[folds[positions] == fold]
        model = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=8000,
            random_state=46042 + int(fold),
        )
        model.fit(features[train], labels[train])
        scores[valid] = model.decision_function(features[valid]).astype(np.float32)
    if not np.isfinite(scores[positions]).all():
        raise ValueError("incomplete embedding OOF scores")
    return scores


def category_scores(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray) -> dict[str, float]:
    return {
        category: f1(labels[categories == category], predictions[categories == category])
        for category in sorted(np.unique(categories))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    args = parser.parse_args()
    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    original = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    data = pd.read_csv(DATA, dtype={"id": str})
    if not np.array_equal(data["id"].astype(str).to_numpy(), ids):
        raise ValueError("data id/order mismatch")
    for name, source in (("qwen3vl", qwen3vl), ("qwen35", original)):
        if not np.array_equal(source["ids"].astype(str), ids):
            raise ValueError(f"{name} id mismatch")

    embedding_path = args.embeddings.resolve()
    embedding_archive = np.load(embedding_path, allow_pickle=False)
    for field, expected in (("ids", ids), ("labels", labels), ("categories", categories)):
        if not np.array_equal(embedding_archive[field].astype(expected.dtype), expected):
            raise ValueError(f"embedding {field} mismatch")
    features = embedding_archive["embeddings"].astype(np.float32)
    if features.shape != (len(labels), 2048) or not np.isfinite(features).all():
        raise ValueError(f"invalid embedding matrix: {features.shape}")
    embedding_scores = crossfit_embedding_scores(features, labels, categories, folds)
    flammable = categories == FLAMMABLE
    embedding_rank = np.zeros(len(labels), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        mask = flammable & (folds == fold)
        embedding_rank[mask] = rank01(embedding_scores[mask])

    parent_paths = [PARENT_QWEN35 / f"fold_{fold}/lora_holdout_predictions.csv" for fold in range(5)]
    parent_logits = load_seed_predictions(parent_paths, base)
    parent_qwen35_rank = fold_category_ranks(parent_logits, folds, categories)
    routed_qwen35_rank = np.where(
        flammable,
        parent_qwen35_rank,
        original["lora_rank"].astype(np.float32),
    )
    base_rank = original["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    matrix = np.column_stack([base_rank, qwen3vl_rank, routed_qwen35_rank])
    baseline, _, baseline_detail = evaluate_locked(matrix, labels, categories, folds)
    frozen = np.load(ROUTE400, allow_pickle=False)
    if not np.array_equal(baseline, frozen["category_routed_nested_predictions"].astype(np.int8)):
        raise ValueError("reconstructed baseline differs from experiment 400")

    parent_score = np.sum(
        matrix * np.asarray([0.15, 0.10, 0.75], dtype=np.float32)[None, :],
        axis=1,
        dtype=np.float32,
    )
    candidate = baseline.copy()
    fold_selection = []
    for fold in sorted(np.unique(folds)):
        train = flammable & (folds != fold)
        valid = flammable & (folds == fold)
        best = None
        for weight in WEIGHTS:
            scores = (1.0 - weight) * parent_score + weight * embedding_rank
            train_f1, threshold = best_threshold(labels[train], scores[train])
            preference = (train_f1, -weight)
            if best is None or preference > best[0]:
                best = (preference, weight, threshold)
        _, weight, threshold = best
        scores = (1.0 - weight) * parent_score + weight * embedding_rank
        candidate[valid] = (scores[valid] >= threshold).astype(np.int8)
        fold_selection.append({
            "fold": int(fold),
            "embedding_weight": float(weight),
            "threshold": float(threshold),
            "validation_f1": f1(labels[valid], candidate[valid]),
        })

    baseline_category = category_scores(labels, baseline, categories)
    candidate_category = category_scores(labels, candidate, categories)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        old = category_scores(labels[mask], baseline[mask], categories[mask])
        new = category_scores(labels[mask], candidate[mask], categories[mask])
        fold_rows.append({
            "fold": int(fold),
            "baseline_macro_f1": float(np.mean(list(old.values()))),
            "candidate_macro_f1": float(np.mean(list(new.values()))),
            "delta_macro_f1": float(np.mean(list(new.values())) - np.mean(list(old.values()))),
            "baseline_category_f1": old,
            "candidate_category_f1": new,
        })
    changed = baseline != candidate
    corrected = int((changed & (candidate == labels) & (baseline != labels)).sum())
    regressed = int((changed & (candidate != labels) & (baseline == labels)).sum())
    positive = flammable & (labels == 1)
    old_fn = int((baseline[positive] == 0).sum())
    new_fn = int((candidate[positive] == 0).sum())
    full_text = data["name"].fillna("").astype(str) + "\n" + data["description"].fillna("").astype(str)
    safety = positive & full_text.str.contains(SAFETY_PATTERN, na=False).to_numpy()
    old_safety_fn = int((baseline[safety] == 0).sum())
    new_safety_fn = int((candidate[safety] == 0).sum())

    guard = pd.read_csv(GUARD, dtype={"id": str, "connected_component": str})
    if not np.array_equal(guard["id"].astype(str).to_numpy(), ids):
        raise ValueError("guard id mismatch")
    connected_audit = connected.audit(
        labels=labels,
        categories=categories,
        folds=folds,
        components=guard["connected_component"].astype(str).to_numpy(),
        safe=guard["safe_for_selection"].astype(bool).to_numpy(),
        baseline=baseline,
        candidate=candidate,
        bootstrap=10000,
        seed=46042,
    )
    baseline_macro = float(np.mean(list(baseline_category.values())))
    candidate_macro = float(np.mean(list(candidate_category.values())))
    macro_delta = candidate_macro - baseline_macro
    flammable_delta = candidate_category[FLAMMABLE] - baseline_category[FLAMMABLE]
    folds_won = sum(row["delta_macro_f1"] > 0 for row in fold_rows)
    connected_safe = connected_audit["connected_safe_rows"]
    gates = {
        "macro_delta_at_least_0_003": macro_delta >= 0.003,
        "flammable_delta_at_least_0_006": flammable_delta >= 0.006,
        "folds_won_at_least_4": folds_won >= 4,
        "corrected_to_regressed_at_least_1_5": corrected >= 1.5 * max(1, regressed),
        "flammable_false_negatives_not_increased": new_fn <= old_fn,
        "safety_union_false_negatives_not_increased": new_safety_fn <= old_safety_fn,
        "connected_bootstrap_probability_at_least_0_90": connected_safe["component_bootstrap"]["probability_delta_positive"] >= 0.90,
    }
    result = {
        "experiment_id": "460",
        "status": "accepted_offline" if all(gates.values()) else "rejected_offline",
        "evaluation_version": "qwen3vl_embedding_transfer_v1",
        "single_changed_factor": "Cross-fitted Qwen3-VL-Embedding-2B multimodal rank added to flammable route",
        "selection_uses_outer_labels": False,
        "baseline": {"category_f1": baseline_category, "macro_f1": baseline_macro},
        "candidate": {"category_f1": candidate_category, "macro_f1": candidate_macro},
        "delta_macro_f1": macro_delta,
        "flammable_delta_f1": flammable_delta,
        "folds_won": folds_won,
        "folds": fold_rows,
        "fold_selection": fold_selection,
        "changed": int(changed.sum()),
        "corrected": corrected,
        "regressed": regressed,
        "flammable_false_negatives": {"baseline": old_fn, "candidate": new_fn, "delta": new_fn - old_fn},
        "safety_union_false_negatives": {"baseline": old_safety_fn, "candidate": new_safety_fn, "delta": new_safety_fn - old_safety_fn},
        "embedding_component": {
            "rows": len(features),
            "dimensions": int(features.shape[1]),
            "flammable_rows": int(flammable.sum()),
            "standalone_nested_f1": float(
                np.mean([
                    f1(
                        labels[flammable & (folds == fold)],
                        (
                            embedding_scores[flammable & (folds == fold)]
                            >= best_threshold(
                                labels[flammable & (folds != fold)],
                                embedding_scores[flammable & (folds != fold)],
                            )[1]
                        ).astype(np.int8),
                    )
                    for fold in sorted(np.unique(folds))
                ])
            ),
        },
        "connected_safe": connected_audit,
        "gates": gates,
        "accepted": all(gates.values()),
        "baseline_detail": baseline_detail,
        "input_sha256": {
            str(DATA.relative_to(ROOT)): sha256(DATA),
            str(GUARD.relative_to(ROOT)): sha256(GUARD),
            str(ROUTE400.relative_to(ROOT)): sha256(ROUTE400),
            str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST),
            str(embedding_path.relative_to(ROOT)): sha256(embedding_path),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acceptance_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "nested_predictions.npz",
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        embedding_scores=embedding_scores,
        embedding_rank=embedding_rank,
        baseline_nested_predictions=baseline,
        candidate_nested_predictions=candidate,
    )
    print(json.dumps({
        "accepted": result["accepted"],
        "macro_delta": macro_delta,
        "flammable_delta": flammable_delta,
        "folds_won": folds_won,
        "corrected": corrected,
        "regressed": regressed,
        "fn_delta": new_fn - old_fn,
        "connected_p": connected_safe["component_bootstrap"]["probability_delta_positive"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
