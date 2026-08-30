from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from qwen35_locked_190_audit import category_scores, evaluate_locked, paired_audit  # noqa: E402
from qwen35_seed_ensemble_cv import (  # noqa: E402
    BASE, QWEN3VL, QWEN35_SEED_A, best_threshold, f1,
    fold_category_ranks, load_seed_predictions,
)


EXPERIMENT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/260_bad_family_diverse_positives/artifacts/seed_42_diverse_positives"
ORIGINAL = ROOT / "experiments/230_qwen35_second_seed/artifacts/seed_42"
ROUTE400 = ROOT / "experiments/400_qwen35_category_routed_adapters/results/routed_predictions.npz"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
FOLDS_FILE = ROOT / "validation/grouped_text_v1/folds.csv"
DATA = ROOT / "research/data.csv"
OUT = EXPERIMENT / "results"
FLAMMABLE = "Легковоспламеняющиеся"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SAFETY_PATTERN = re.compile(
    r"(?iu)\b(?:зажигалк|спич|огнив|факел|свеч|горелк|фейерверк|салют|петард|бенгал|пиротех|дымогенератор|свеч\w*\s*фонтан|фонтан\w*\s+для\s+торт|угол|уголь|дров|брик|розжиг)\w*\b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_intermediate(path: Path, alpha: float, ids, labels, categories, folds) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"id": str})
    positions = np.flatnonzero(categories == FLAMMABLE)
    if not np.array_equal(frame.id.astype(str).to_numpy(), ids[positions]):
        raise ValueError(f"alpha {alpha} id/order mismatch")
    if not np.array_equal(frame.label.astype(np.int8).to_numpy(), labels[positions]):
        raise ValueError(f"alpha {alpha} label mismatch")
    if not np.array_equal(frame.fold.astype(np.int8).to_numpy(), folds[positions]):
        raise ValueError(f"alpha {alpha} fold mismatch")
    if set(frame.category.astype(str)) != {FLAMMABLE}:
        raise ValueError(f"alpha {alpha} category mismatch")
    values = frame.lora_score.astype(np.float32).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"alpha {alpha} non-finite score")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-25", type=Path, required=True)
    parser.add_argument("--alpha-50", type=Path, required=True)
    parser.add_argument("--alpha-75", type=Path, required=True)
    args = parser.parse_args()
    paths = {0.25: args.alpha_25, 0.5: args.alpha_50, 0.75: args.alpha_75}

    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    original_cache = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    for cache in (qwen3vl, original_cache):
        if not np.array_equal(ids, cache["ids"].astype(str)):
            raise ValueError("base component id mismatch")

    original_paths = [ORIGINAL / f"fold_{fold}/extracted/lora_holdout_predictions.csv" for fold in range(5)]
    parent_paths = [PARENT / f"fold_{fold}/lora_holdout_predictions.csv" for fold in range(5)]
    original_logits = load_seed_predictions(original_paths, base)
    parent_logits = load_seed_predictions(parent_paths, base)
    flammable_positions = np.flatnonzero(categories == FLAMMABLE)
    logits = {0.0: original_logits, 1.0: parent_logits}
    for alpha, path in paths.items():
        values = original_logits.copy()
        values[flammable_positions] = load_intermediate(
            path, alpha, ids, labels, categories, folds
        )
        logits[alpha] = values

    base_rank = original_cache["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    original_rank = original_cache["lora_rank"].astype(np.float32)
    original_nested, _, _ = evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, original_rank]), labels, categories, folds
    )
    parent_rank = fold_category_ranks(parent_logits, folds, categories)
    parent_nested, _, _ = evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, parent_rank]), labels, categories, folds
    )
    baseline = np.where(categories == FLAMMABLE, parent_nested, original_nested).astype(np.int8)
    frozen = np.load(ROUTE400, allow_pickle=False)
    if not np.array_equal(frozen["ids"].astype(str), ids):
        raise ValueError("experiment 400 id mismatch")
    if not np.array_equal(frozen["category_routed_nested_predictions"].astype(np.int8), baseline):
        raise ValueError("experiment 400 reconstruction mismatch")

    ranks = {alpha: fold_category_ranks(values, folds, categories) for alpha, values in logits.items()}
    fused = {
        alpha: 0.15 * base_rank + 0.10 * qwen3vl_rank + 0.75 * rank
        for alpha, rank in ranks.items()
    }
    candidate = original_nested.copy()
    fold_selection = []
    for fold in range(5):
        train = (categories == FLAMMABLE) & (folds != fold)
        valid = (categories == FLAMMABLE) & (folds == fold)
        trials = []
        for alpha in ALPHAS:
            train_f1, threshold = best_threshold(labels[train], fused[alpha][train])
            trials.append({"alpha_260": alpha, "train_f1": train_f1, "threshold": threshold})
        selected = sorted(trials, key=lambda item: (-item["train_f1"], item["alpha_260"]))[0]
        predictions = (fused[selected["alpha_260"]][valid] >= selected["threshold"]).astype(np.int8)
        candidate[valid] = predictions
        fold_selection.append({
            "fold": fold,
            "selected_alpha_260": selected["alpha_260"],
            "threshold": selected["threshold"],
            "train_f1": selected["train_f1"],
            "validation_f1": f1(labels[valid], predictions),
            "trials": trials,
        })

    fold_frame = pd.read_csv(FOLDS_FILE, dtype={"id": str})
    if not np.array_equal(fold_frame.id.astype(str).to_numpy(), ids):
        raise ValueError("fold file id mismatch")
    audit = paired_audit(
        name="nested_alpha_adapter_soup",
        labels=labels,
        categories=categories,
        folds=folds,
        group_hashes=fold_frame.group_hash.astype(str).to_numpy(),
        baseline=baseline,
        candidate=candidate,
        bootstrap=10000,
        seed=430,
    )
    guard = pd.read_csv(GUARD, dtype={"id": str, "connected_component": str})
    if not np.array_equal(guard.id.astype(str).to_numpy(), ids):
        raise ValueError("connected guard id mismatch")
    safe = guard.safe_for_selection.astype(bool).to_numpy()
    safe_audit = paired_audit(
        name="nested_alpha_adapter_soup_connected_safe",
        labels=labels[safe],
        categories=categories[safe],
        folds=folds[safe],
        group_hashes=guard.connected_component.astype(str).to_numpy()[safe],
        baseline=baseline[safe],
        candidate=candidate[safe],
        bootstrap=10000,
        seed=431,
    )

    data = pd.read_csv(DATA, dtype={"id": str})
    if not np.array_equal(data.id.astype(str).to_numpy(), ids):
        raise ValueError("data id mismatch")
    text = data.name.fillna("").astype(str) + "\n" + data.description.fillna("").astype(str)
    positive = (categories == FLAMMABLE) & (labels == 1)
    safety = positive & text.str.contains(SAFETY_PATTERN, na=False).to_numpy()
    fn = {
        "flammable": {
            "baseline": int((baseline[positive] == 0).sum()),
            "candidate": int((candidate[positive] == 0).sum()),
        },
        "safety_union": {
            "positive_rows": int(safety.sum()),
            "baseline": int((baseline[safety] == 0).sum()),
            "candidate": int((candidate[safety] == 0).sum()),
        },
    }
    fn["flammable"]["delta"] = fn["flammable"]["candidate"] - fn["flammable"]["baseline"]
    fn["safety_union"]["delta"] = fn["safety_union"]["candidate"] - fn["safety_union"]["baseline"]
    ratio = float("inf") if audit["regressed"] == 0 else audit["corrected"] / audit["regressed"]
    gates = {
        "macro_delta_at_least_0_003": audit["delta_macro_f1"] >= 0.003,
        "flammable_delta_at_least_0_006": audit["category_delta"][FLAMMABLE] >= 0.006,
        "fold_wins_at_least_4": audit["folds_won"] >= 4,
        "flammable_false_negatives_not_increased": fn["flammable"]["delta"] <= 0,
        "safety_union_false_negatives_not_increased": fn["safety_union"]["delta"] <= 0,
        "corrected_to_regressed_at_least_1_5": ratio >= 1.5,
        "connected_delta_positive": safe_audit["delta_macro_f1"] > 0,
        "connected_bootstrap_probability_at_least_0_90": safe_audit["group_bootstrap"]["probability_delta_positive"] >= 0.90,
    }
    production_trials = []
    mask = categories == FLAMMABLE
    for alpha in ALPHAS:
        value, threshold = best_threshold(labels[mask], fused[alpha][mask])
        production_trials.append({"alpha_260": alpha, "oof_f1": value, "threshold": threshold})
    production = sorted(production_trials, key=lambda item: (-item["oof_f1"], item["alpha_260"]))[0]
    result = {
        "experiment_id": "430",
        "evaluation_version": "locked_190_category_routing_soup_v1",
        "selection_manifest_sha256": sha256(EXPERIMENT / "analysis/selection_manifest.json"),
        "alpha_grid": list(ALPHAS),
        "fold_selection": fold_selection,
        "nested_audit": audit,
        "connected_safe_audit": safe_audit,
        "false_negatives": fn,
        "corrected_to_regressed_ratio": ratio,
        "gates": gates,
        "accepted_before_prior_replay": all(gates.values()),
        "production_alpha_diagnostic": {"selected": production, "trials": production_trials},
        "input_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path)
            for path in paths.values()
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
        baseline=baseline,
        candidate=candidate,
    )
    print(json.dumps({
        "macro": audit["candidate_macro_f1"],
        "delta": audit["delta_macro_f1"],
        "flammable_delta": audit["category_delta"][FLAMMABLE],
        "folds_won": audit["folds_won"],
        "corrected": audit["corrected"],
        "regressed": audit["regressed"],
        "fn": fn,
        "connected_delta": safe_audit["delta_macro_f1"],
        "connected_p": safe_audit["group_bootstrap"]["probability_delta_positive"],
        "selected_alphas": [item["selected_alpha_260"] for item in fold_selection],
        "accepted_before_prior_replay": all(gates.values()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
