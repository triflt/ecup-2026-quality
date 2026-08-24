from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY_PATH = ROOT / "experiments/635_span_head_full140_integration/evaluate.py"
FLAMMABLE = "Легковоспламеняющиеся"
SCREEN_FOLDS = (0, 3)
FULL_FOLDS = (0, 1, 2, 3, 4)
FROZEN_BUNDLE_SHA256 = "f2a7c40bad91170c515c1461f6e04c11e4957694f69a815c8e2b8fe80656ec3d"
FROZEN_CONTRACT_SHA256 = "1771dceda26996529b9f70f46a97653c2d29afa099983260c3b40a62fca518b0"
FROZEN_REGISTRY_SHA256 = "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae"


def load_legacy():
    spec = importlib.util.spec_from_file_location("exp635_for_680", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    if positives == 0 or not np.isfinite(scores).all():
        raise ValueError("AP requires positive labels and finite scores")
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    tp = np.cumsum(ranked_labels == 1)
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(scores) - 1]
    precision = tp[ends] / (ends + 1)
    recall = tp[ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def load_scores(paths: list[Path], *, bundle: dict[str, np.ndarray], folds_scope: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    records: list[dict[str, Any]] = []
    hashes: list[str] = []
    for path in paths:
        hashes.append(sha256_file(path))
        records.extend(
            row
            for line in path.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line)).get("category") == FLAMMABLE
        )
    ids = bundle["ids"].astype(str)
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    positions = np.flatnonzero(np.isin(folds, folds_scope) & (categories == FLAMMABLE))
    expected = set(ids[positions])
    by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        if set(row) & {"label", "target", "gold", "sealed", "public"}:
            raise ValueError("prediction supervision is forbidden")
        row_id = str(row["id"])
        if row_id in by_id:
            raise ValueError("duplicate prediction ID")
        by_id[row_id] = row
    if set(by_id) != expected:
        raise ValueError("predictions do not exactly cover flammable validation rows")
    output = bundle["qwen35_original_logit"].astype(np.float64).copy()
    for position in positions:
        row = by_id[ids[position]]
        score = float(row["score"])
        if int(row["fold"]) != int(folds[position]) or str(row["category"]) != FLAMMABLE:
            raise ValueError("prediction binding mismatch")
        if not np.isfinite(score) or int(row["prediction"]) != int(score >= 0.0):
            raise ValueError("invalid prediction score")
        output[position] = score
    return output, hashes


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite evaluation")
    folds_scope = SCREEN_FOLDS if args.stage == "screen" else FULL_FOLDS
    if len(args.candidate_score) != len(folds_scope) or len(args.control_score) != len(folds_scope):
        raise ValueError("score-file count differs from stage")
    if sha256_file(args.bundle) != FROZEN_BUNDLE_SHA256 or sha256_file(args.replay_contract) != FROZEN_CONTRACT_SHA256 or sha256_file(args.registry) != FROZEN_REGISTRY_SHA256:
        raise ValueError("frozen input checksum mismatch")
    spec = LEGACY.load_spec()
    LEGACY.verify_source_recipe(spec)
    contract = LEGACY.verify_replay_contract(path=args.replay_contract, bundle_path=args.bundle, registry_path=args.registry, spec=spec)
    registry = LEGACY._load_registry(args.registry, spec=spec)
    bundle = LEGACY.load_replay_bundle(path=args.bundle, contract=contract, registry=registry)
    candidate_logits, candidate_hashes = load_scores(args.candidate_score, bundle=bundle, folds_scope=folds_scope)
    control_logits, control_hashes = load_scores(args.control_score, bundle=bundle, folds_scope=folds_scope)
    labels = registry["label"].astype(np.int8).to_numpy()
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    components = bundle["semantic_components"].astype(str)
    selected = np.isin(folds, folds_scope) & (categories == FLAMMABLE)
    fold_ap: dict[str, Any] = {}
    for fold in folds_scope:
        local = selected & (folds == fold)
        before = average_precision(labels[local], control_logits[local])
        after = average_precision(labels[local], candidate_logits[local])
        fold_ap[str(fold)] = {"control": before, "candidate": after, "delta": after - before}
    baseline_prob = LEGACY.sigmoid(bundle["qwen35_original_logit"])
    candidate_prob = baseline_prob.copy()
    candidate_prob[categories == FLAMMABLE] = LEGACY.sigmoid(candidate_logits[categories == FLAMMABLE])
    baseline_pred, _ = LEGACY.route_predictions(bundle=bundle, qwen_probability=baseline_prob, route=spec["route"])
    candidate_pred, _ = LEGACY.route_predictions(bundle=bundle, qwen_probability=candidate_prob, route=spec["route"])
    metrics = LEGACY._metrics(labels=labels, categories=categories, folds=folds, components=components, baseline=baseline_pred, candidate=candidate_pred, selected_folds=folds_scope, bootstrap=spec["bootstrap"] if args.stage == "full" else None)
    ratio_ok = metrics["corrected"] > 0 if metrics["regressed"] == 0 else metrics["corrected_to_regressed"] >= 1.5
    ap_deltas = [fold_ap[str(fold)]["delta"] for fold in folds_scope]
    common = {
        "bad_route_byte_identical": metrics["categories"]["БАД"]["delta"] == 0.0,
        "corrected_to_regressed_at_least_1_5": ratio_ok,
        "flammable_false_negatives_do_not_increase": metrics["false_negatives"]["flammable"]["delta"] <= 0,
    }
    if args.stage == "screen":
        gates = common | {
            "both_folds_ap_positive": all(delta > 0 for delta in ap_deltas),
            "mean_ap_delta_at_least_0_005": float(np.mean(ap_deltas)) >= 0.005,
            "both_folds_macro_positive": metrics["winning_folds"] == 2,
            "mean_macro_delta_at_least_0_0015": metrics["mean_fold_delta"] >= 0.0015,
            "flammable_f1_does_not_drop": metrics["categories"][FLAMMABLE]["delta"] >= 0.0,
        }
    else:
        gates = common | {
            "at_least_four_of_five_ap_wins": sum(delta > 0 for delta in ap_deltas) >= 4,
            "mean_ap_delta_at_least_0_005": float(np.mean(ap_deltas)) >= 0.005,
            "all_confirmation_folds_macro_positive": all(metrics["folds"][str(f)]["delta"] > 0 for f in (1, 2, 4)),
            "at_least_four_of_five_macro_wins": metrics["winning_folds"] >= 4,
            "macro_delta_at_least_0_006": metrics["macro_delta"] >= 0.006,
            "flammable_f1_delta_at_least_0_012": metrics["categories"][FLAMMABLE]["delta"] >= 0.012,
            "bootstrap_probability_at_least_0_90": metrics["component_bootstrap"]["probability_delta_positive"] >= 0.90,
        }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "experiment_id": "680",
        "stage": args.stage,
        "changed_factor": "remove_bad_training_occurrences",
        "fold_flammable_average_precision": fold_ap,
        "mean_flammable_average_precision_delta": float(np.mean(ap_deltas)),
        "production_metrics": metrics,
        "gates": gates,
        "passed": passed,
        "decision": ("OPEN_CONFIRMATION" if passed else "REJECT_AT_SCREEN") if args.stage == "screen" else ("ACCEPT_FOR_REFIT" if passed else "REJECT_FULL"),
        "candidate_prediction_sha256": candidate_hashes,
        "control_prediction_sha256": control_hashes,
        "public_used": False,
        "sealed_rows": 0,
        "threshold_tuned": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "full"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replay-contract", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-score", type=Path, action="append", required=True)
    parser.add_argument("--control-score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2, sort_keys=True))
