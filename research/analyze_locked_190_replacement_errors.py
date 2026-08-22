from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/data.csv"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
PREDICTIONS = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
IMAGE_MANIFEST = ROOT / "research/multi_image_manifest.tsv.gz"

PATTERNS = {
    "bad_explicit": r"(?iu)(?:\bбад\b|биологически\s+активн\w*\s+добавк|dietary\s+supplement|food\s+supplement)",
    "sports": r"(?iu)(?:спортпит|спортивн\w*\s+питан|\bbcaa\b|\bпротеин|гейнер|креатин|аминокислот)",
    "vitamin": r"(?iu)(?:витамин|минерал|магни|цинк|желез|омега[-\s]?3|коэнзим|пробиотик)",
    "medicine": r"(?iu)(?:лекарств|таблетк|капсул|сироп|мазь|спрей|препарат)",
    "gas": r"(?iu)\b(?:газ|пропан|бутан|баллон|картридж)\w*\b",
    "burner": r"(?iu)\b(?:горелк|плит|печ|примус)\w*\b",
    "lighter": r"(?iu)\b(?:зажигалк|спич|огнив|факел)\w*\b",
    "fuel_liquid": r"(?iu)(?:\b(?:топлив|бензин|керосин)\w*\b|жидкост\w*\s+для\s+розжиг)",
    "candle": r"(?iu)\b(?:свеч|воск|парафин)\w*\b",
    "kit": r"(?iu)\b(?:комплект|набор|входит|поставк)\w*\b",
    "charcoal": r"(?iu)\b(?:угол|уголь|брик|дров)\w*\b",
}


def normalize(value: str) -> str:
    value = html.unescape(str(value or "")).lower()
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\W+", " ", value, flags=re.U).strip()


def f1(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None


def cohort_row(frame: pd.DataFrame, cohort: str, mask: np.ndarray) -> dict:
    local = frame.loc[mask]
    old_ok = local.label.to_numpy() == local.baseline_prediction.to_numpy()
    new_ok = local.label.to_numpy() == local.candidate_prediction.to_numpy()
    old_f1 = f1(local.label.to_numpy(), local.baseline_prediction.to_numpy())
    new_f1 = f1(local.label.to_numpy(), local.candidate_prediction.to_numpy())
    return {
        "cohort": cohort,
        "rows": len(local),
        "positives": int(local.label.sum()),
        "positive_rate": float(local.label.mean()) if len(local) else None,
        "families": int(local.group_hash.nunique()),
        "baseline_f1": old_f1,
        "candidate_f1": new_f1,
        "delta_f1": None if old_f1 is None or new_f1 is None else new_f1 - old_f1,
        "baseline_errors": int((~old_ok).sum()),
        "candidate_errors": int((~new_ok).sum()),
        "corrected": int((~old_ok & new_ok).sum()),
        "regressed": int((old_ok & ~new_ok).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="exp260")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/260_bad_family_diverse_positives/analysis/locked_190_errors",
    )
    args = parser.parse_args()

    frame = pd.read_csv(DATA, dtype={"id": str})
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["text"] = frame.name + "\n" + frame.description
    frame["normalized_text"] = frame.text.map(normalize)
    folds = pd.read_csv(FOLDS, dtype={"id": str})[["id", "fold", "group_hash"]]
    frame = frame.merge(folds, on="id", validate="one_to_one")

    predictions = np.load(PREDICTIONS, allow_pickle=True)
    if not np.array_equal(frame.id.to_numpy(), predictions["ids"].astype(str)):
        raise ValueError("prediction id order mismatch")
    candidate_key = f"{args.candidate}_nested_predictions"
    if candidate_key not in predictions:
        raise ValueError(f"unknown candidate: {args.candidate}")
    frame["baseline_prediction"] = predictions["baseline_nested_predictions"].astype(np.int8)
    frame["candidate_prediction"] = predictions[candidate_key].astype(np.int8)

    manifest = pd.read_csv(IMAGE_MANIFEST, sep="\t", compression="gzip", dtype={"id": str})
    manifest["image_count"] = manifest.image_urls.map(lambda value: len(json.loads(value)))
    frame = frame.merge(manifest[["id", "image_count"]], on="id", validate="one_to_one")

    group = frame.groupby(["category", "group_hash"], sort=False)
    frame["family_size"] = group.id.transform("size")
    frame["family_label_count"] = group.label.transform("nunique")
    frame["family_kind"] = np.select(
        [frame.family_size == 1, frame.family_label_count > 1],
        ["singleton", "conflicting_repeat"],
        default="consistent_repeat",
    )
    frame["description_chars"] = frame.description.str.len()
    frame["description_bin"] = pd.qcut(
        frame.description_chars.rank(method="first"),
        4,
        labels=["Q1_short", "Q2", "Q3", "Q4_long"],
    ).astype(str)
    frame["transition"] = np.select(
        [
            (frame.label != frame.baseline_prediction) & (frame.label == frame.candidate_prediction),
            (frame.label == frame.baseline_prediction) & (frame.label != frame.candidate_prediction),
            (frame.label != frame.baseline_prediction) & (frame.label != frame.candidate_prediction),
        ],
        ["corrected", "regressed", "wrong_in_both"],
        default="right_in_both",
    )
    for name, pattern in PATTERNS.items():
        frame[name] = frame.text.str.contains(pattern, regex=True, na=False)

    rows = []
    for category in sorted(frame.category.unique()):
        category_mask = (frame.category == category).to_numpy()
        rows.append(cohort_row(frame, f"{category}:all", category_mask))
        for label in (0, 1):
            rows.append(cohort_row(frame, f"{category}:label_{label}", category_mask & (frame.label == label)))
        for fold in sorted(frame.fold.unique()):
            rows.append(cohort_row(frame, f"{category}:fold_{fold}", category_mask & (frame.fold == fold)))
        for kind in ("singleton", "consistent_repeat", "conflicting_repeat"):
            rows.append(cohort_row(frame, f"{category}:family:{kind}", category_mask & (frame.family_kind == kind)))
        for count in sorted(frame.image_count.unique()):
            rows.append(cohort_row(frame, f"{category}:images:{count}", category_mask & (frame.image_count == count)))
        for length_bin in ("Q1_short", "Q2", "Q3", "Q4_long"):
            rows.append(cohort_row(frame, f"{category}:description:{length_bin}", category_mask & (frame.description_bin == length_bin)))
        signals = ("bad_explicit", "sports", "vitamin", "medicine") if category == "БАД" else (
            "gas", "burner", "lighter", "fuel_liquid", "candle", "kit", "charcoal"
        )
        for signal in signals:
            rows.append(cohort_row(frame, f"{category}:signal:{signal}", category_mask & frame[signal].to_numpy()))

    cohorts = pd.DataFrame(rows)
    family_summary = group.agg(
        rows=("id", "size"),
        positives=("label", "sum"),
        label_count=("label", "nunique"),
        baseline_errors=("baseline_prediction", lambda value: int((value != frame.loc[value.index, "label"]).sum())),
        candidate_errors=("candidate_prediction", lambda value: int((value != frame.loc[value.index, "label"]).sum())),
        example_name=("name", "first"),
    ).reset_index()
    family_summary["error_delta"] = family_summary.candidate_errors - family_summary.baseline_errors

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    columns = [
        "id", "category", "label", "fold", "group_hash", "family_size", "family_kind",
        "image_count", "description_chars", "baseline_prediction", "candidate_prediction",
        "transition", "name", "description",
    ]
    frame.loc[frame.transition.isin(["corrected", "regressed"]), columns].sort_values(
        ["category", "transition", "label", "fold"]
    ).to_csv(output / "changed_decisions.csv", index=False)
    frame.loc[frame.transition == "wrong_in_both", columns].sort_values(
        ["category", "label", "fold"]
    ).to_csv(output / "remaining_shared_errors.csv", index=False)
    cohorts.to_csv(output / "cohort_metrics.csv", index=False)
    family_summary.sort_values(["error_delta", "rows"], ascending=[False, False]).to_csv(
        output / "family_summary.csv", index=False
    )

    transitions = frame.transition.value_counts().astype(int).to_dict()
    strongest = cohorts.loc[
        (cohorts.rows >= 25) & cohorts.delta_f1.notna()
    ].sort_values("delta_f1")
    report = {
        "candidate": args.candidate,
        "rows": len(frame),
        "transitions": transitions,
        "by_category": {
            category: cohort_row(frame, category, (frame.category == category).to_numpy())
            for category in sorted(frame.category.unique())
        },
        "largest_cohort_regressions": strongest.head(10).to_dict("records"),
        "largest_cohort_gains": strongest.tail(10).sort_values("delta_f1", ascending=False).to_dict("records"),
        "outputs": [
            "changed_decisions.csv", "remaining_shared_errors.csv", "cohort_metrics.csv", "family_summary.csv"
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
