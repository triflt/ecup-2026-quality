from __future__ import annotations

import hashlib
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_predictions(fold: int, name: str, expected_ids: np.ndarray) -> np.ndarray:
    path = ARTIFACTS / f"fold_{fold}" / name
    frame = pd.read_csv(path, dtype={"id": str})
    if not np.array_equal(frame.id.astype(str).to_numpy(), expected_ids):
        raise ValueError(f"{name} fold {fold} id mismatch")
    values = frame.lora_score.astype(np.float32).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{name} fold {fold} non-finite score")
    return values


def category_macro(
    labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray
) -> dict[str, float]:
    scores = category_scores(labels, predictions, categories)
    scores["Macro"] = float(np.mean(list(scores.values())))
    return scores


def main() -> None:
    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    seed_a = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    for cache in (qwen3vl, seed_a):
        if not np.array_equal(cache["ids"].astype(str), ids):
            raise ValueError("component id mismatch")
    base_rank = seed_a["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)

    frozen = np.load(ROUTE400, allow_pickle=False)
    if not np.array_equal(frozen["ids"].astype(str), ids):
        raise ValueError("experiment 400 id mismatch")
    route400 = frozen["category_routed_nested_predictions"].astype(np.int8)
    source430 = json.loads(SOURCE430.read_text())
    thresholds = {
        int(item["fold"]): float(item["threshold"])
        for item in source430["fold_selection"]
        if int(item["fold"]) in SCREEN_FOLDS
    }
    if set(thresholds) != set(SCREEN_FOLDS):
        raise ValueError("missing frozen thresholds")

    pure = route400.copy()
    soup = route400.copy()
    fold_reports = []
    for fold in SCREEN_FOLDS:
        all_positions = np.flatnonzero(folds == fold)
        positions = np.flatnonzero((folds == fold) & (categories == FLAMMABLE))
        pure_all = read_predictions(
            fold, "lora_holdout_predictions.csv", ids[all_positions]
        )
        local_flammable = categories[all_positions] == FLAMMABLE
        pure_values = pure_all[local_flammable]
        soup_values = read_predictions(
            fold, "soup_holdout_predictions.csv", ids[positions]
        )
        if len(pure_values) != len(soup_values):
            raise ValueError(f"fold {fold} flammable row mismatch")
        pure_fused = (
            0.15 * base_rank[positions]
            + 0.10 * qwen3vl_rank[positions]
            + 0.75 * rank01(pure_values)
        )
        soup_fused = (
            0.15 * base_rank[positions]
            + 0.10 * qwen3vl_rank[positions]
            + 0.75 * rank01(soup_values)
        )
        threshold = thresholds[fold]
        pure[positions] = (pure_fused >= threshold).astype(np.int8)
        soup[positions] = (soup_fused >= threshold).astype(np.int8)
        mask = folds == fold
        pure_scores = category_macro(labels[mask], pure[mask], categories[mask])
        soup_scores = category_macro(labels[mask], soup[mask], categories[mask])
        fold_reports.append({
            "fold": fold,
            "threshold": threshold,
            "pure_macro_f1": pure_scores["Macro"],
            "soup_macro_f1": soup_scores["Macro"],
            "delta_macro_f1": soup_scores["Macro"] - pure_scores["Macro"],
            "pure_flammable_f1": pure_scores[FLAMMABLE],
            "soup_flammable_f1": soup_scores[FLAMMABLE],
        })

    screen = np.isin(folds, SCREEN_FOLDS)
    fold_frame = pd.read_csv(FOLDS_FILE, dtype={"id": str})
    if not np.array_equal(fold_frame.id.astype(str).to_numpy(), ids):
        raise ValueError("fold manifest id mismatch")
    audit = paired_audit(
        name="fixed_alpha_seed31415_two_fold_screen",
        labels=labels[screen],
        categories=categories[screen],
        folds=folds[screen],
        group_hashes=fold_frame.group_hash.astype(str).to_numpy()[screen],
        baseline=pure[screen],
        candidate=soup[screen],
        bootstrap=10000,
        seed=470,
    )
    guard = pd.read_csv(GUARD, dtype={"id": str, "connected_component": str})
    if not np.array_equal(guard.id.astype(str).to_numpy(), ids):
        raise ValueError("connected guard id mismatch")
    connected = screen & guard.safe_for_selection.astype(bool).to_numpy()
    connected_audit = paired_audit(
        name="fixed_alpha_seed31415_two_fold_connected_safe",
        labels=labels[connected],
        categories=categories[connected],
        folds=folds[connected],
        group_hashes=guard.connected_component.astype(str).to_numpy()[connected],
        baseline=pure[connected],
        candidate=soup[connected],
        bootstrap=10000,
        seed=471,
    )

    data = pd.read_csv(DATA, dtype={"id": str})
    if not np.array_equal(data.id.astype(str).to_numpy(), ids):
        raise ValueError("data id mismatch")
    text = data.name.fillna("").astype(str) + "\n" + data.description.fillna("").astype(str)
    positive = screen & (categories == FLAMMABLE) & (labels == 1)
    safety = positive & text.str.contains(SAFETY_PATTERN, na=False).to_numpy()
    false_negatives = {
        "flammable": {
            "pure": int((pure[positive] == 0).sum()),
            "soup": int((soup[positive] == 0).sum()),
        },
        "safety_union": {
            "positive_rows": int(safety.sum()),
            "pure": int((pure[safety] == 0).sum()),
            "soup": int((soup[safety] == 0).sum()),
        },
    }
    for item in false_negatives.values():
        item["delta"] = item["soup"] - item["pure"]
    gates = {
        "both_folds_positive": all(item["delta_macro_f1"] > 0 for item in fold_reports),
        "mean_macro_delta_positive": bool(
            np.mean([item["delta_macro_f1"] for item in fold_reports]) > 0
        ),
        "flammable_false_negatives_not_increased": false_negatives["flammable"]["delta"] <= 0,
        "safety_union_false_negatives_not_increased": false_negatives["safety_union"]["delta"] <= 0,
        "corrected_greater_than_regressed": audit["corrected"] > audit["regressed"],
    }
    result = {
        "experiment_id": "470",
        "evaluation_version": "qwen35_adapter_soup_seed_repeat_screen_v1",
        "seed": 31415,
        "folds": list(SCREEN_FOLDS),
        "alpha_specialist": 0.5,
        "weights": [0.15, 0.10, 0.75],
        "threshold_source_sha256": sha256(SOURCE430),
        "fold_reports": fold_reports,
        "screen_audit": audit,
        "connected_safe_audit": connected_audit,
        "false_negatives": false_negatives,
        "gates": gates,
        "accepted": all(gates.values()),
        "input_sha256": {
            f"fold_{fold}_{name}": sha256(ARTIFACTS / f"fold_{fold}" / name)
            for fold in SCREEN_FOLDS
            for name in ("lora_holdout_predictions.csv", "soup_holdout_predictions.csv")
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "screen_predictions.npz",
        ids=ids[screen],
        labels=labels[screen],
        categories=categories[screen],
        folds=folds[screen],
        pure_predictions=pure[screen],
        soup_predictions=soup[screen],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
