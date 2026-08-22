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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from qwen35_locked_190_audit import evaluate_locked  # noqa: E402
from qwen35_seed_ensemble_cv import (  # noqa: E402
    BASE,
    QWEN3VL,
    QWEN35_SEED_A,
    f1,
    fold_category_ranks,
    load_seed_predictions,
)


FLAMMABLE = "Легковоспламеняющиеся"
SCREEN_FOLDS = (0, 3)
PARENT = ROOT / "experiments/260_bad_family_diverse_positives/artifacts/seed_42_diverse_positives"
ROUTE400 = ROOT / "experiments/400_qwen35_category_routed_adapters/results/routed_predictions.npz"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
DATA = ROOT / "research/data.csv"
OUT = Path(__file__).resolve().parent / "results"
SAFETY_PATTERN = re.compile(
    r"(?iu)\b(?:зажигалк|спич|огнив|факел|свеч|горелк|фейерверк|салют|петард|бенгал|пиротех|дымогенератор|свеч\w*\s*фонтан|фонтан\w*\s+для\s+торт|угол|уголь|дров|брик|розжиг)\w*\b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_scores(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray) -> dict[str, float]:
    return {
        category: f1(labels[categories == category], predictions[categories == category])
        for category in sorted(np.unique(categories))
    }


def fold_audit(
    *,
    fold: int,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    safe: np.ndarray,
    safety_union: np.ndarray,
) -> dict[str, object]:
    mask = folds == fold
    old = category_scores(labels[mask], baseline[mask], categories[mask])
    new = category_scores(labels[mask], candidate[mask], categories[mask])
    old_macro = float(np.mean(list(old.values())))
    new_macro = float(np.mean(list(new.values())))
    changed = mask & (baseline != candidate)
    corrected = int((changed & (candidate == labels) & (baseline != labels)).sum())
    regressed = int((changed & (candidate != labels) & (baseline == labels)).sum())
    flammable_positive = mask & (categories == FLAMMABLE) & (labels == 1)
    old_fn = int((baseline[flammable_positive] == 0).sum())
    new_fn = int((candidate[flammable_positive] == 0).sum())
    safety = mask & safety_union
    old_safety_fn = int((baseline[safety] == 0).sum())
    new_safety_fn = int((candidate[safety] == 0).sum())

    safe_mask = mask & safe
    safe_old = category_scores(labels[safe_mask], baseline[safe_mask], categories[safe_mask])
    safe_new = category_scores(labels[safe_mask], candidate[safe_mask], categories[safe_mask])
    safe_delta = float(np.mean(list(safe_new.values())) - np.mean(list(safe_old.values())))
    gates = {
        "routed_macro_delta_positive": new_macro - old_macro > 0,
        "flammable_false_negatives_not_increased": new_fn - old_fn <= 0,
        "corrected_exceeds_regressed": corrected > regressed,
        "safety_union_false_negatives_not_increased": new_safety_fn - old_safety_fn <= 0,
    }
    return {
        "fold": fold,
        "baseline_category_f1": old,
        "candidate_category_f1": new,
        "baseline_macro_f1": old_macro,
        "candidate_macro_f1": new_macro,
        "delta_macro_f1": new_macro - old_macro,
        "changed": int(changed.sum()),
        "corrected": corrected,
        "regressed": regressed,
        "flammable_false_negatives": {
            "baseline": old_fn,
            "candidate": new_fn,
            "delta": new_fn - old_fn,
        },
        "safety_union_false_negatives": {
            "positive_rows": int(safety.sum()),
            "baseline": old_safety_fn,
            "candidate": new_safety_fn,
            "delta": new_safety_fn - old_safety_fn,
        },
        "connected_safe_delta_macro_f1": safe_delta,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-0", type=Path, required=True)
    parser.add_argument("--fold-3", type=Path, required=True)
    args = parser.parse_args()
    candidate_paths = {0: args.fold_0, 3: args.fold_3}

    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    original = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    if not np.array_equal(ids, qwen3vl["ids"].astype(str)) or not np.array_equal(ids, original["ids"].astype(str)):
        raise ValueError("base component id mismatch")

    parent_paths = [PARENT / f"fold_{fold}/lora_holdout_predictions.csv" for fold in range(5)]
    parent_logits = load_seed_predictions(parent_paths, base)
    hybrid_logits = parent_logits.copy()
    for fold, path in candidate_paths.items():
        frame = pd.read_csv(path, dtype={"id": str})
        if "category" not in frame or "fold" not in frame:
            raise ValueError(f"fold {fold} continuation file lacks category/fold columns")
        frame = frame[
            (frame["category"].astype(str) == FLAMMABLE)
            & (frame["fold"].astype(int) == fold)
        ].copy()
        expected = np.flatnonzero((folds == fold) & (categories == FLAMMABLE))
        expected_ids = ids[expected]
        if not np.array_equal(frame["id"].astype(str).to_numpy(), expected_ids):
            raise ValueError(f"fold {fold} continuation prediction id/order mismatch")
        if not np.all(frame["fold"].astype(int).to_numpy() == fold):
            raise ValueError(f"fold {fold} continuation file contains another fold")
        if set(frame["category"].astype(str)) != {FLAMMABLE}:
            raise ValueError(f"fold {fold} continuation file contains another category")
        if not np.array_equal(frame["label"].astype(np.int8).to_numpy(), labels[expected]):
            raise ValueError(f"fold {fold} continuation label mismatch")
        scores = frame["lora_score"].astype(np.float32).to_numpy()
        if not np.isfinite(scores).all():
            raise ValueError(f"fold {fold} continuation scores are not finite")
        hybrid_logits[expected] = scores

    parent_rank = fold_category_ranks(parent_logits, folds, categories)
    hybrid_rank = fold_category_ranks(hybrid_logits, folds, categories)
    base_rank = original["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    original_rank = original["lora_rank"].astype(np.float32)
    original_nested, _, _ = evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, original_rank]), labels, categories, folds
    )
    parent_nested, _, parent_detail = evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, parent_rank]), labels, categories, folds
    )
    hybrid_nested, _, hybrid_detail = evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, hybrid_rank]), labels, categories, folds
    )
    route400 = np.where(categories == FLAMMABLE, parent_nested, original_nested).astype(np.int8)
    route420 = np.where(categories == FLAMMABLE, hybrid_nested, original_nested).astype(np.int8)

    frozen = np.load(ROUTE400, allow_pickle=False)
    if not np.array_equal(frozen["ids"].astype(str), ids):
        raise ValueError("experiment 400 routed id mismatch")
    if not np.array_equal(frozen["category_routed_nested_predictions"].astype(np.int8), route400):
        raise ValueError("reconstructed route400 predictions differ from frozen artifact")

    guard = pd.read_csv(GUARD, dtype={"id": str})
    if not np.array_equal(guard["id"].astype(str).to_numpy(), ids):
        raise ValueError("connected guard id mismatch")
    safe = guard["safe_for_selection"].astype(bool).to_numpy()
    data = pd.read_csv(DATA, dtype={"id": str})
    if not np.array_equal(data["id"].astype(str).to_numpy(), ids):
        raise ValueError("data id mismatch")
    text = data["name"].fillna("").astype(str) + "\n" + data["description"].fillna("").astype(str)
    safety_union = (
        (categories == FLAMMABLE)
        & (labels == 1)
        & text.str.contains(SAFETY_PATTERN, na=False).to_numpy()
    )

    folds_report = [
        fold_audit(
            fold=fold,
            labels=labels,
            categories=categories,
            folds=folds,
            baseline=route400,
            candidate=route420,
            safe=safe,
            safety_union=safety_union,
        )
        for fold in SCREEN_FOLDS
    ]
    result = {
        "experiment_id": "420",
        "evaluation_version": "flammable_recall_two_fold_reject_screen_v1",
        "purpose": "Reject-only screen; cannot accept or package the candidate",
        "screen_folds": list(SCREEN_FOLDS),
        "threshold_protocol": "Hybrid nested threshold: continuation OOF ranks on folds 0 and 3, unchanged parent-260 OOF ranks on folds 1, 2 and 4; each tested fold threshold excludes that fold",
        "weights": {"robust_base": 0.15, "qwen3vl": 0.10, "qwen35": 0.75},
        "selection_uses_screen_labels": False,
        "folds": folds_report,
        "screen_passed": all(item["passed"] for item in folds_report),
        "full_five_fold_cycle_allowed": all(item["passed"] for item in folds_report),
        "parent_locked_detail": parent_detail[FLAMMABLE],
        "hybrid_locked_detail": hybrid_detail[FLAMMABLE],
        "input_sha256": {
            str(args.fold_0): sha256(args.fold_0),
            str(args.fold_3): sha256(args.fold_3),
            str(ROUTE400.relative_to(ROOT)): sha256(ROUTE400),
            str(GUARD.relative_to(ROOT)): sha256(GUARD),
            str(DATA.relative_to(ROOT)): sha256(DATA),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "screen_predictions.npz",
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        route400_nested_predictions=route400,
        route420_screen_nested_predictions=route420,
        parent_rank=parent_rank,
        hybrid_rank=hybrid_rank,
    )
    print(json.dumps({
        "screen_passed": result["screen_passed"],
        "folds": [
            {
                "fold": item["fold"],
                "delta": item["delta_macro_f1"],
                "corrected": item["corrected"],
                "regressed": item["regressed"],
                "fn_delta": item["flammable_false_negatives"]["delta"],
                "safety_fn_delta": item["safety_union_false_negatives"]["delta"],
                "passed": item["passed"],
            }
            for item in folds_report
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
