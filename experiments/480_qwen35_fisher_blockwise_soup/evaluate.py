from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from qwen35_locked_190_audit import category_scores, paired_audit
from qwen35_seed_ensemble_cv import QWEN3VL, QWEN35_SEED_A, rank01

EXPERIMENT = Path(__file__).resolve().parent
BASE = ROOT / "research/oof-cache-extracted/oof_scores.npz"
ROUTE400 = ROOT / "experiments/400_qwen35_category_routed_adapters/results/routed_predictions.npz"
SOURCE430 = ROOT / "experiments/430_qwen35_flammable_adapter_soup/results/acceptance_audit.json"
BASELINE470 = ROOT / "experiments/470_qwen35_adapter_soup_seed_repeat/artifacts/seed_31415"
FOLDS_FILE = ROOT / "validation/grouped_text_v1/folds.csv"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
DATA = ROOT / "research/data.csv"
ARTIFACTS = EXPERIMENT / "artifacts/seed_31415"
OUT = EXPERIMENT / "results"
SCREEN_FOLDS = (0, 3)
FLAMMABLE = "Легковоспламеняющиеся"
SAFETY_PATTERN = re.compile(
    r"(?iu)\b(?:зажигалк|спич|огнив|факел|свеч|горелк|фейерверк|салют|петард|бенгал|пиротех|дымогенератор|свеч\w*\s*фонтан|фонтан\w*\s+для\s+торт|угол|уголь|дров|брик|розжиг)\w*\b"
)


def read_scores(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"id": str})
    if not np.array_equal(frame.id.astype(str).to_numpy(), expected_ids):
        raise ValueError(f"id mismatch: {path}")
    values = frame.lora_score.astype(np.float32).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite score: {path}")
    return values


def macro(labels, predictions, categories):
    scores = category_scores(labels, predictions, categories)
    scores["Macro"] = float(np.mean(list(scores.values())))
    return scores


def rank_correlation(left: dict[str, float], right: dict[str, float]) -> float:
    if set(left) != set(right):
        raise ValueError("coefficient block mismatch")
    keys = sorted(left)
    a = pd.Series([left[key] for key in keys]).rank(method="average").to_numpy()
    b = pd.Series([right[key] for key in keys]).rank(method="average").to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    seed_a = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    base_rank = seed_a["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    frozen = np.load(ROUTE400, allow_pickle=False)
    baseline = frozen["category_routed_nested_predictions"].astype(np.int8).copy()
    candidate = baseline.copy()
    source = json.loads(SOURCE430.read_text())
    thresholds = {int(item["fold"]): float(item["threshold"]) for item in source["fold_selection"]}

    fold_reports = []
    coefficient_reports = []
    coefficient_maps = []
    for fold in SCREEN_FOLDS:
        positions = np.flatnonzero((folds == fold) & (categories == FLAMMABLE))
        global_scores = read_scores(
            BASELINE470 / f"fold_{fold}/soup_holdout_predictions.csv", ids[positions]
        )
        fisher_scores = read_scores(
            ARTIFACTS / f"fold_{fold}/fisher_holdout_predictions.csv", ids[positions]
        )
        global_fused = (
            0.15 * base_rank[positions]
            + 0.10 * qwen3vl_rank[positions]
            + 0.75 * rank01(global_scores)
        )
        fisher_fused = (
            0.15 * base_rank[positions]
            + 0.10 * qwen3vl_rank[positions]
            + 0.75 * rank01(fisher_scores)
        )
        threshold = thresholds[fold]
        baseline[positions] = (global_fused >= threshold).astype(np.int8)
        candidate[positions] = (fisher_fused >= threshold).astype(np.int8)
        mask = folds == fold
        left = macro(labels[mask], baseline[mask], categories[mask])
        right = macro(labels[mask], candidate[mask], categories[mask])
        fold_reports.append(
            {
                "fold": fold,
                "threshold": threshold,
                "baseline_macro_f1": left["Macro"],
                "candidate_macro_f1": right["Macro"],
                "delta_macro_f1": right["Macro"] - left["Macro"],
                "baseline_flammable_f1": left[FLAMMABLE],
                "candidate_flammable_f1": right[FLAMMABLE],
            }
        )
        fisher = json.loads((ARTIFACTS / f"fold_{fold}/fisher_block_alphas.json").read_text())
        values = {key: float(value) for key, value in fisher["block_alphas"].items()}
        coefficient_maps.append(values)
        coefficient_reports.append(
            {
                "fold": fold,
                "blocks": len(values),
                "min": min(values.values()),
                "median": float(np.median(list(values.values()))),
                "max": max(values.values()),
                "selective_fraction": float(
                    np.mean([abs(value - 0.5) >= 0.1 for value in values.values()])
                ),
            }
        )

    screen = np.isin(folds, SCREEN_FOLDS)
    fold_frame = pd.read_csv(FOLDS_FILE, dtype={"id": str})
    audit = paired_audit(
        name="fisher_blockwise_vs_global_half_seed31415",
        labels=labels[screen],
        categories=categories[screen],
        folds=folds[screen],
        group_hashes=fold_frame.group_hash.astype(str).to_numpy()[screen],
        baseline=baseline[screen],
        candidate=candidate[screen],
        bootstrap=10000,
        seed=480,
    )
    guard = pd.read_csv(GUARD, dtype={"id": str, "connected_component": str})
    connected = screen & guard.safe_for_selection.astype(bool).to_numpy()
    connected_audit = paired_audit(
        name="fisher_blockwise_vs_global_half_connected_safe",
        labels=labels[connected],
        categories=categories[connected],
        folds=folds[connected],
        group_hashes=guard.connected_component.astype(str).to_numpy()[connected],
        baseline=baseline[connected],
        candidate=candidate[connected],
        bootstrap=10000,
        seed=481,
    )
    data = pd.read_csv(DATA, dtype={"id": str})
    text = data.name.fillna("").astype(str) + "\n" + data.description.fillna("").astype(str)
    positive = screen & (categories == FLAMMABLE) & (labels == 1)
    safety = positive & text.str.contains(SAFETY_PATTERN, na=False).to_numpy()
    false_negatives = {
        "flammable": {
            "baseline": int((baseline[positive] == 0).sum()),
            "candidate": int((candidate[positive] == 0).sum()),
        },
        "safety_union": {
            "baseline": int((baseline[safety] == 0).sum()),
            "candidate": int((candidate[safety] == 0).sum()),
        },
    }
    for item in false_negatives.values():
        item["delta"] = item["candidate"] - item["baseline"]
    correlation = rank_correlation(*coefficient_maps)
    gates = {
        "both_folds_positive": all(item["delta_macro_f1"] > 0 for item in fold_reports),
        "mean_macro_delta_at_least_0_001": audit["delta_macro_f1"] >= 0.001,
        "flammable_false_negatives_not_increased": false_negatives["flammable"]["delta"] <= 0,
        "safety_union_false_negatives_not_increased": false_negatives["safety_union"]["delta"] <= 0,
        "corrected_greater_than_regressed": audit["corrected"] > audit["regressed"],
        "selective_fraction_at_least_0_25": all(
            item["selective_fraction"] >= 0.25 for item in coefficient_reports
        ),
        "alpha_rank_correlation_at_least_0_50": correlation >= 0.50,
    }
    result = {
        "experiment_id": "480",
        "evaluation_version": "qwen35_fisher_blockwise_soup_screen_v1",
        "seed": 31415,
        "folds": list(SCREEN_FOLDS),
        "baseline": "global_alpha_0.50_same_seed",
        "fold_reports": fold_reports,
        "coefficient_reports": coefficient_reports,
        "coefficient_rank_correlation": correlation,
        "screen_audit": audit,
        "connected_safe_audit": connected_audit,
        "false_negatives": false_negatives,
        "gates": gates,
        "accepted": all(gates.values()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(
        OUT / "screen_predictions.npz",
        ids=ids[screen],
        labels=labels[screen],
        categories=categories[screen],
        folds=folds[screen],
        baseline_predictions=baseline[screen],
        candidate_predictions=candidate[screen],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
