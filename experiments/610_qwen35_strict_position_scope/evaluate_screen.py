from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from position_protocol import EXPERIMENT_ID, PARSER_VERSION, SCREEN_FOLDS, canonical_sha256
from position_shared import sha256_file

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
WEIGHTS = {
    BAD: (0.50, 0.25, 0.25),
    FLAMMABLE: (0.15, 0.10, 0.75),
}
SAFETY_PATTERN = re.compile(
    r"(?iu)\b(?:зажигалк|спич|огнив|факел|свеч|горелк|фейерверк|салют|"
    r"петард|бенгал|пиротех|дымогенератор|свеч\w*\s*фонтан|"
    r"фонтан\w*\s+для\s+торт|угол|уголь|дров|брик|розжиг)\w*\b"
)


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return result


def fold_category_ranks(values: np.ndarray, folds: np.ndarray, categories: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float32)
    for fold in range(5):
        for category in sorted(np.unique(categories)):
            mask = (folds == fold) & (categories == category)
            result[mask] = rank01(values[mask])
    return result


def best_threshold(labels: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.quantile(values, np.linspace(0.002, 0.998, 700)))
    scores = np.asarray([f1(labels, values >= threshold) for threshold in candidates])
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def _load_five(specifications: list[list[str]], *, expected_component: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw_fold, raw_dir in specifications:
        fold = int(raw_fold)
        path = Path(raw_dir).resolve()
        if fold not in range(5) or fold in result:
            raise ValueError("baseline fold directories must uniquely cover 0..4")
        contract = json.loads((path / "output_contract.runtime.json").read_text(encoding="utf-8"))
        if (
            contract.get("experiment_id") != "600"
            or contract.get("component") != expected_component
            or contract.get("outer_fold") != fold
            or contract.get("decision") != "GO"
            or contract.get("sealed_rows_in_predictions") != 0
        ):
            raise ValueError(f"invalid experiment-600 {expected_component} fold {fold}")
        result[fold] = path
    if set(result) != set(range(5)):
        raise ValueError(f"incomplete {expected_component} baseline grid")
    return result


def _combine(paths: dict[int, Path], ids: np.ndarray, folds: np.ndarray) -> np.ndarray:
    scores = np.empty(len(ids), dtype=np.float32)
    for fold, directory in paths.items():
        frame = pd.read_csv(directory / "lora_holdout_predictions.csv", dtype={"id": str})
        expected = ids[folds == fold]
        if not np.array_equal(frame["id"].astype(str).to_numpy(), expected):
            raise ValueError(f"baseline fold {fold} prediction IDs/order mismatch")
        scores[folds == fold] = frame["lora_score"].to_numpy(np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("baseline logits contain non-finite values")
    return scores


def _calibrate_baseline(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    robust_rank: np.ndarray,
    visual_rank: np.ndarray,
    qwen_rank: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[str, int], float]]:
    predictions = np.zeros(len(labels), dtype=np.int8)
    thresholds: dict[tuple[str, int], float] = {}
    for category in sorted(np.unique(categories)):
        w_base, w_visual, w_qwen = WEIGHTS[category]
        fused = w_base * robust_rank + w_visual * visual_rank + w_qwen * qwen_rank
        for fold in range(5):
            donor = (categories == category) & (folds != fold)
            valid = (categories == category) & (folds == fold)
            threshold, _ = best_threshold(labels[donor], fused[donor])
            predictions[valid] = (fused[valid] >= threshold).astype(np.int8)
            thresholds[(category, fold)] = threshold
    return predictions, thresholds


def _fold_report(
    *,
    fold: int,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    components: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    safety: np.ndarray,
) -> dict:
    mask = folds == fold
    old = {category: f1(labels[mask & (categories == category)], baseline[mask & (categories == category)]) for category in sorted(np.unique(categories))}
    new = {category: f1(labels[mask & (categories == category)], candidate[mask & (categories == category)]) for category in sorted(np.unique(categories))}
    changed = mask & (baseline != candidate)
    corrected = int((changed & (baseline != labels) & (candidate == labels)).sum())
    regressed = int((changed & (baseline == labels) & (candidate != labels)).sum())
    positive = mask & (categories == FLAMMABLE) & (labels == 1)
    old_fn = int((baseline[positive] == 0).sum())
    new_fn = int((candidate[positive] == 0).sum())
    safety_mask = mask & safety
    old_safety_fn = int((baseline[safety_mask] == 0).sum())
    new_safety_fn = int((candidate[safety_mask] == 0).sum())
    changed_components = Counter(components[changed].astype(str))
    old_macro, new_macro = float(np.mean(list(old.values()))), float(np.mean(list(new.values())))
    gates = {
        "macro_delta_positive": new_macro > old_macro,
        "flammable_delta_positive": new[FLAMMABLE] > old[FLAMMABLE],
        "bad_predictions_byte_identical": bool(np.array_equal(baseline[mask & (categories == BAD)], candidate[mask & (categories == BAD)])),
        "flammable_false_negatives_not_increased": new_fn <= old_fn,
        "safety_false_negatives_not_increased": new_safety_fn <= old_safety_fn,
        "corrected_exceeds_regressed": corrected > regressed,
        "at_least_five_changed_decisions": int(changed.sum()) >= 5,
        "gain_not_single_component_over_10_rows": max(changed_components.values(), default=0) <= 10,
    }
    return {
        "fold": fold,
        "baseline_macro_f1": old_macro,
        "candidate_macro_f1": new_macro,
        "delta_macro_f1": new_macro - old_macro,
        "baseline_category_f1": old,
        "candidate_category_f1": new,
        "category_delta": {key: new[key] - old[key] for key in old},
        "changed": int(changed.sum()),
        "corrected": corrected,
        "regressed": regressed,
        "flammable_false_negative_delta": new_fn - old_fn,
        "safety_false_negative_delta": new_safety_fn - old_safety_fn,
        "largest_changed_component_rows": max(changed_components.values(), default=0),
        "gates": gates,
        "passed": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the semantic-v3 strict position screen.")
    parser.add_argument("--visual-components", required=True, type=Path)
    parser.add_argument("--visual-contract", required=True, type=Path)
    parser.add_argument("--development-data", required=True, type=Path)
    parser.add_argument("--baseline-original", action="append", nargs=2, default=[], metavar=("FOLD", "DIR"))
    parser.add_argument("--baseline-specialist", action="append", nargs=2, default=[], metavar=("FOLD", "DIR"))
    parser.add_argument("--candidate", action="append", nargs=2, default=[], metavar=("FOLD", "DIR"))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite screen evaluation")
    original_dirs = _load_five(args.baseline_original, expected_component="original")
    specialist_dirs = _load_five(args.baseline_specialist, expected_component="specialist")
    candidate_dirs = {int(fold): Path(directory).resolve() for fold, directory in args.candidate}
    if set(candidate_dirs) != set(SCREEN_FOLDS):
        raise ValueError("candidate must contain exactly screen folds 0 and 3")
    visual_contract = json.loads(args.visual_contract.read_text(encoding="utf-8"))
    expected_visual = {
        "version": "semantic_v3_visual_base_components_v1",
        "status": "complete",
        "development_rows": 11_118,
        "sealed_rows_in_outputs": 0,
        "qwen35_component_present": False,
    }
    visual_mismatches = {
        key: {"expected": value, "actual": visual_contract.get(key)}
        for key, value in expected_visual.items()
        if visual_contract.get(key) != value
    }
    expected_visual_sha = visual_contract.get("output_sha256", {}).get(
        "semantic_v3_visual_base_components.npz"
    )
    if visual_mismatches or expected_visual_sha != sha256_file(args.visual_components):
        raise ValueError(
            f"semantic-v3 visual component contract mismatch: {visual_mismatches}"
        )
    visual = np.load(args.visual_components, allow_pickle=False)
    ids = visual["ids"].astype(str)
    labels = visual["labels"].astype(np.int8)
    categories = visual["categories"].astype(str)
    folds = visual["folds"].astype(np.int8)
    components = visual["semantic_components"].astype(str)
    data = pd.read_csv(args.development_data, dtype={"id": str}).fillna("")
    if not np.array_equal(data["id"].astype(str).to_numpy(), ids):
        raise ValueError("development data IDs/order differ from visual components")
    original_rank = fold_category_ranks(_combine(original_dirs, ids, folds), folds, categories)
    specialist_rank = fold_category_ranks(_combine(specialist_dirs, ids, folds), folds, categories)
    routed_qwen_rank = np.where(categories == FLAMMABLE, specialist_rank, original_rank)
    robust_rank = visual["robust_base_rank"].astype(np.float32)
    visual_rank = visual["qwen3vl_rank"].astype(np.float32)
    baseline, thresholds = _calibrate_baseline(
        labels=labels,
        categories=categories,
        folds=folds,
        robust_rank=robust_rank,
        visual_rank=visual_rank,
        qwen_rank=routed_qwen_rank,
    )
    candidate_rank = routed_qwen_rank.copy()
    contracts = {}
    for fold, directory in sorted(candidate_dirs.items()):
        contract_path = directory / "position_output_contract.runtime.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if (
            contract.get("experiment_id") != EXPERIMENT_ID
            or contract.get("parser_version") != PARSER_VERSION
            or contract.get("outer_fold") != fold
            or contract.get("decision") != "GO"
            or contract.get("sealed_rows_in_predictions") != 0
        ):
            raise ValueError(f"candidate fold {fold} output contract mismatch")
        frame = pd.read_csv(directory / "lora_holdout_predictions.csv", dtype={"id": str})
        expected = ids[folds == fold]
        if not np.array_equal(frame["id"].astype(str).to_numpy(), expected):
            raise ValueError(f"candidate fold {fold} IDs/order mismatch")
        local_scores = frame["lora_score"].to_numpy(np.float32)
        if not np.isfinite(local_scores).all():
            raise ValueError("candidate logits contain non-finite values")
        local_categories = categories[folds == fold]
        for category in sorted(np.unique(categories)):
            local = local_categories == category
            candidate_rank[np.flatnonzero(folds == fold)[local]] = rank01(local_scores[local])
        contracts[str(fold)] = sha256_file(contract_path)
    candidate = baseline.copy()
    for fold in SCREEN_FOLDS:
        local = folds == fold
        for category in sorted(np.unique(categories)):
            mask = local & (categories == category)
            if category == BAD:
                candidate[mask] = baseline[mask]
                continue
            w_base, w_visual, w_qwen = WEIGHTS[category]
            fused = w_base * robust_rank[mask] + w_visual * visual_rank[mask] + w_qwen * candidate_rank[mask]
            candidate[mask] = (fused >= thresholds[(category, fold)]).astype(np.int8)
    text = data["name"].astype(str) + "\n" + data["description"].astype(str)
    safety = (categories == FLAMMABLE) & (labels == 1) & text.str.contains(SAFETY_PATTERN, na=False).to_numpy()
    folds_report = [
        _fold_report(
            fold=fold,
            labels=labels,
            categories=categories,
            folds=folds,
            components=components,
            baseline=baseline,
            candidate=candidate,
            safety=safety,
        )
        for fold in SCREEN_FOLDS
    ]
    mean_delta = float(np.mean([item["delta_macro_f1"] for item in folds_report]))
    screen_passed = all(item["passed"] for item in folds_report) and mean_delta >= 0.001
    report = {
        "experiment_id": EXPERIMENT_ID,
        "evaluation_version": "semantic_v3_route400_strict_position_screen_v1",
        "purpose": "reject-only; passing cannot accept the hypothesis",
        "weights": {key: {"robust": value[0], "qwen3vl": value[1], "qwen35": value[2]} for key, value in WEIGHTS.items()},
        "thresholds_fit_on_baseline_donor_folds_only": True,
        "candidate_changes_thresholds": False,
        "candidate_changes_inference": False,
        "candidate_contract_sha256": contracts,
        "folds": folds_report,
        "mean_screen_delta_macro_f1": mean_delta,
        "required_mean_delta_macro_f1": 0.001,
        "screen_passed": screen_passed,
        "full_cycle_allowed": screen_passed,
        "sealed_holdout_used": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "screen_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "screen_predictions.npz",
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        baseline_predictions=baseline,
        candidate_predictions=candidate,
    )
    print(json.dumps({"screen_passed": screen_passed, "mean_delta": mean_delta}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
