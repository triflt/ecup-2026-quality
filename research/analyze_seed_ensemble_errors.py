from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/data.csv"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
BASELINE = ROOT / "research/qwen3vl-qwen35-5fold-nested-fusion.npz"
ENSEMBLE = ROOT / "experiments/230_qwen35_second_seed/results/seed_ensemble_report.npz"
ARTIFACTS = ROOT / "experiments/230_qwen35_second_seed/artifacts"
OUTPUT = ROOT / "experiments/230_qwen35_second_seed/analysis"


PATTERNS = {
    "gas": r"(?iu)\b(?:газ|пропан|бутан|баллон|картридж)\w*\b",
    "burner": r"(?iu)\b(?:горелк|плит|печ|примус)\w*\b",
    "lighter": r"(?iu)\b(?:зажигалк|спич|огнив|факел)\w*\b",
    "fuel_liquid": r"(?iu)\b(?:топлив|бензин|керосин|жидкост\w*\s+для\s+розжиг)\w*\b",
    "candle": r"(?iu)\b(?:свеч|воск|парафин)\w*\b",
    "kit": r"(?iu)\b(?:комплект|набор|входит|поставк)\w*\b",
    "charcoal": r"(?iu)\b(?:угол|уголь|брик|дров)\w*\b",
    "bad_explicit": r"(?iu)(?:\bбад\b|биологически\s+активн\w*\s+добавк|dietary\s+supplement)",
    "sports": r"(?iu)\b(?:протеин|гейнер|креатин|bcaa|спортпит|аминокислот)\w*\b",
}


def load_raw_seed(seed: str) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        path = ARTIFACTS / f"seed_{seed}" / f"fold_{fold}" / "extracted/lora_holdout_predictions.csv"
        local = pd.read_csv(path, dtype={"id": str})
        if set(local["fold"].astype(int)) != {fold}:
            raise ValueError(f"fold mismatch in {path}")
        frames.append(local[["id", "lora_score"]])
    result = pd.concat(frames, ignore_index=True)
    if result.id.duplicated().any():
        raise ValueError(f"duplicate ids for seed {seed}")
    return result.rename(columns={"lora_score": f"seed_{seed}_logit"})


def error_count(frame: pd.DataFrame, prediction: str) -> int:
    return int((frame.label.to_numpy() != frame[prediction].to_numpy()).sum())


def cohort_row(frame: pd.DataFrame, name: str, mask: np.ndarray) -> dict:
    local = frame.loc[mask]
    baseline_ok = local.label.to_numpy() == local.baseline_prediction.to_numpy()
    candidate_ok = local.label.to_numpy() == local.candidate_prediction.to_numpy()
    return {
        "cohort": name,
        "rows": len(local),
        "positive": int(local.label.sum()),
        "families": int(local.group_hash.nunique()),
        "baseline_errors": int((~baseline_ok).sum()),
        "candidate_errors": int((~candidate_ok).sum()),
        "corrected": int((~baseline_ok & candidate_ok).sum()),
        "regressed": int((baseline_ok & ~candidate_ok).sum()),
    }


def main() -> None:
    frame = pd.read_csv(DATA, dtype={"id": str})
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["text"] = frame["name"] + "\n" + frame["description"]
    folds = pd.read_csv(FOLDS, dtype={"id": str})[["id", "fold", "group_hash"]]
    frame = frame.merge(folds, on="id", validate="one_to_one")

    baseline = np.load(BASELINE, allow_pickle=True)
    ensemble = np.load(ENSEMBLE, allow_pickle=True)
    ids = frame.id.to_numpy()
    for name, values in (("baseline", baseline), ("ensemble", ensemble)):
        if not np.array_equal(ids, values["ids"].astype(str)):
            raise ValueError(f"{name} id order mismatch")
    frame["baseline_prediction"] = baseline["nested_predictions"].astype(np.int8)
    frame["candidate_prediction"] = ensemble["probability_nested_predictions"].astype(np.int8)
    frame["seed_42_rank"] = ensemble["qwen35_seed42_rank"].astype(np.float32)
    frame["seed_31415_rank"] = ensemble["qwen35_seed31415_rank"].astype(np.float32)
    frame = frame.merge(load_raw_seed("42"), on="id", validate="one_to_one")
    frame = frame.merge(load_raw_seed("31415"), on="id", validate="one_to_one")
    for seed in ("42", "31415"):
        values = np.clip(frame[f"seed_{seed}_logit"].to_numpy(), -40, 40)
        frame[f"seed_{seed}_probability"] = 1.0 / (1.0 + np.exp(-values))
    frame["seed_mean_probability"] = (
        frame.seed_42_probability + frame.seed_31415_probability
    ) / 2
    frame["seed_probability_disagreement"] = np.abs(
        frame.seed_42_probability - frame.seed_31415_probability
    )

    group = frame.groupby(["category", "group_hash"], sort=False)
    frame["family_size"] = group.id.transform("size")
    frame["family_positive_rate"] = group.label.transform("mean")
    frame["family_label_count"] = group.label.transform("nunique")
    frame["family_kind"] = np.select(
        [frame.family_size == 1, frame.family_label_count > 1],
        ["singleton", "conflicting_repeat"],
        default="consistent_repeat",
    )
    frame["transition"] = np.select(
        [
            (frame.label != frame.baseline_prediction) & (frame.label == frame.candidate_prediction),
            (frame.label == frame.baseline_prediction) & (frame.label != frame.candidate_prediction),
            (frame.label != frame.baseline_prediction) & (frame.label != frame.candidate_prediction),
        ],
        ["corrected", "regressed", "wrong_in_both"],
        default="right_in_both",
    )

    cohort_rows = []
    for category in sorted(frame.category.unique()):
        category_mask = frame.category.to_numpy() == category
        cohort_rows.append(cohort_row(frame, f"{category}:all", category_mask))
        for fold in range(5):
            cohort_rows.append(
                cohort_row(frame, f"{category}:fold_{fold}", category_mask & (frame.fold == fold))
            )
        for family_kind in ("singleton", "consistent_repeat", "conflicting_repeat"):
            cohort_rows.append(
                cohort_row(
                    frame,
                    f"{category}:family:{family_kind}",
                    category_mask & (frame.family_kind == family_kind),
                )
            )
        signals = ("bad_explicit", "sports") if category == "БАД" else (
            "gas", "burner", "lighter", "fuel_liquid", "candle", "kit", "charcoal"
        )
        for signal in signals:
            signal_mask = frame.text.str.contains(PATTERNS[signal], regex=True, na=False).to_numpy()
            cohort_rows.append(
                cohort_row(frame, f"{category}:signal:{signal}", category_mask & signal_mask)
            )
    cohorts = pd.DataFrame(cohort_rows)

    family_summary = group.agg(
        rows=("id", "size"),
        positive_rate=("label", "mean"),
        label_count=("label", "nunique"),
        baseline_errors=("baseline_prediction", lambda value: int((value != frame.loc[value.index, "label"]).sum())),
        candidate_errors=("candidate_prediction", lambda value: int((value != frame.loc[value.index, "label"]).sum())),
        mean_seed_disagreement=("seed_probability_disagreement", "mean"),
        example_name=("name", "first"),
    ).reset_index()
    family_summary["error_delta"] = family_summary.candidate_errors - family_summary.baseline_errors

    columns = [
        "id", "category", "label", "fold", "group_hash", "family_size", "family_kind",
        "baseline_prediction", "candidate_prediction", "transition",
        "seed_42_logit", "seed_31415_logit", "seed_42_probability",
        "seed_31415_probability", "seed_mean_probability", "seed_probability_disagreement",
        "name", "description",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.loc[frame.transition.isin(["corrected", "regressed"]), columns].sort_values(
        ["category", "transition", "seed_probability_disagreement"], ascending=[True, True, False]
    ).to_csv(OUTPUT / "changed_decisions.csv", index=False)
    frame.loc[frame.candidate_prediction != frame.label, columns].sort_values(
        ["category", "label", "seed_probability_disagreement"], ascending=[True, False, False]
    ).to_csv(OUTPUT / "remaining_errors.csv", index=False)
    cohorts.to_csv(OUTPUT / "cohort_metrics.csv", index=False)
    family_summary.sort_values(
        ["error_delta", "mean_seed_disagreement"], ascending=[False, False]
    ).to_csv(OUTPUT / "family_summary.csv", index=False)

    report = {
        "rows": len(frame),
        "baseline_errors": error_count(frame, "baseline_prediction"),
        "candidate_errors": error_count(frame, "candidate_prediction"),
        "transitions": frame.transition.value_counts().astype(int).to_dict(),
        "by_category": {
            category: {
                "rows": int(mask.sum()),
                "positive": int(frame.loc[mask, "label"].sum()),
                "baseline_errors": error_count(frame.loc[mask], "baseline_prediction"),
                "candidate_errors": error_count(frame.loc[mask], "candidate_prediction"),
                "corrected": int((frame.loc[mask, "transition"] == "corrected").sum()),
                "regressed": int((frame.loc[mask, "transition"] == "regressed").sum()),
            }
            for category in sorted(frame.category.unique())
            for mask in [frame.category == category]
        },
        "flammable_positive_families": int(
            frame.loc[(frame.category == "Легковоспламеняющиеся") & (frame.label == 1), "group_hash"].nunique()
        ),
        "output_files": [
            "changed_decisions.csv", "remaining_errors.csv", "cohort_metrics.csv", "family_summary.csv"
        ],
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
