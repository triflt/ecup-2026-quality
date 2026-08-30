from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from assemble_oof import best_threshold, fold_category_percentile_rank

DATA = Path("/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/data.csv")
FOLDS = Path(__file__).resolve().parents[2] / "validation/grouped_text_v1/folds.csv"


def evaluate_predictions(
    pred: pd.DataFrame, data: pd.DataFrame, folds: pd.DataFrame
) -> dict:
    if set(pred.columns) != {"id", "fold", "score"} or pred.empty:
        raise ValueError("prediction schema mismatch")
    pred = pred.copy()
    pred["id"] = pred["id"].astype(str)
    if pred["id"].astype(str).duplicated().any():
        raise ValueError("duplicate prediction IDs")
    if not np.isfinite(pred["score"].to_numpy(np.float64)).all():
        raise ValueError("prediction scores must be finite")
    data = data.copy()
    folds = folds.copy()
    data["id"] = data["id"].astype(str)
    folds["id"] = folds["id"].astype(str)
    truth = data[["id", "category", "label"]].merge(folds[["id", "fold"]], on="id")
    available_folds = sorted(pred["fold"].astype(int).unique().tolist())
    if len(available_folds) < 2 or any(fold not in range(5) for fold in available_folds):
        raise ValueError("nested evaluation requires at least two valid folds")
    expected_ids = set(truth.loc[truth["fold"].isin(available_folds), "id"])
    if set(pred["id"]) != expected_ids:
        raise ValueError("prediction coverage mismatch for available folds")
    merged = pred.merge(truth, on=["id", "fold"], validate="one_to_one")
    merged["rank_score"] = fold_category_percentile_rank(
        merged["score"].to_numpy(np.float64),
        merged["fold"].to_numpy(),
        merged["category"].astype(str).to_numpy(),
    )
    result = {
        "schema_version": "exp697_screen_evaluation_v2",
        "folds": available_folds,
        "coverage_rows": int(len(merged)),
        "full_five_fold_oof": len(available_folds) == 5,
        "score_calibration": "label_blind_percentile_rank_within_fold_and_category_average_ties",
        "raw_scores_preserved_in_fold_predictions": True,
        "categories": {},
    }
    category_f1 = []
    for category, local in merged.groupby("category"):
        labels = local["label"].to_numpy(np.int8)
        scores = local["rank_score"].to_numpy(np.float64)
        fold_scores = {}
        predictions = np.zeros(len(local), dtype=np.int8)
        local_folds = local["fold"].to_numpy()
        for fold in available_folds:
            valid = local_folds == fold
            donor = ~valid
            threshold = best_threshold(labels[donor], scores[donor]) if donor.any() else 0.0
            predictions[valid] = scores[valid] >= threshold
            fold_scores[str(fold)] = {
                "rank_threshold": threshold,
                "f1": float(f1_score(labels[valid], predictions[valid])),
                "rows": int(valid.sum()),
            }
        f1 = float(f1_score(labels, predictions))
        result["categories"][category] = {"nested_f1": f1, "folds": fold_scores}
        category_f1.append(f1)
    result["nested_macro_f1"] = float(np.mean(category_f1))
    if len(available_folds) == 5:
        result["solution_140_nested_macro_f1"] = 0.9118425205786493
        result["delta_vs_140"] = (
            result["nested_macro_f1"] - result["solution_140_nested_macro_f1"]
        )
        result["decision_scope"] = "full_grouped_text_v1_oof"
    else:
        result["solution_140_nested_macro_f1"] = None
        result["delta_vs_140"] = None
        result["decision_scope"] = "screen_only_not_comparable_to_full_140"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.predictions:
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    result = evaluate_predictions(
        pd.DataFrame(rows),
        pd.read_csv(DATA, dtype={"id": str}),
        pd.read_csv(FOLDS, dtype={"id": str}),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
