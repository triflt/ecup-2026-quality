"""Evaluate experiment 679 inside the frozen production-fusion replay.

The standalone 4B evaluator is necessary but not sufficient: this evaluator
replaces only the flammable Qwen3.5 signal in the checksum-locked semantic-v3
replay of the production route.  BAD remains byte-identical to the control.
"""

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
FROZEN_BUNDLE_SHA256 = "f2a7c40bad91170c515c1461f6e04c11e4957694f69a815c8e2b8fe80656ec3d"
FROZEN_CONTRACT_SHA256 = "1771dceda26996529b9f70f46a97653c2d29afa099983260c3b40a62fca518b0"
FROZEN_REGISTRY_SHA256 = "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae"
SCREEN_FOLDS = (0, 3)
FULL_FOLDS = (0, 1, 2, 3, 4)
FLAMMABLE = "Легковоспламеняющиеся"
REQUIRED_PREDICTION_FIELDS = {
    "global_index",
    "id",
    "fold",
    "category",
    "score",
    "prediction",
}
FORBIDDEN_PREDICTION_FIELDS = {"label", "target", "gold", "answer", "sealed", "public"}


def load_legacy():
    spec = importlib.util.spec_from_file_location("experiment_635_frozen_evaluator", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen evaluator: {LEGACY_PATH}")
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


def load_candidate_scores(
    paths: list[Path], *, bundle: dict[str, np.ndarray], folds_scope: tuple[int, ...]
) -> tuple[np.ndarray, list[str]]:
    records: list[dict[str, Any]] = []
    hashes: list[str] = []
    for path in paths:
        hashes.append(sha256_file(path))
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    if any(
        not REQUIRED_PREDICTION_FIELDS <= set(row)
        or FORBIDDEN_PREDICTION_FIELDS.intersection(row)
        for row in records
    ):
        raise ValueError("candidate prediction schema or supervision mismatch")
    ids = bundle["ids"].astype(str)
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    expected_positions = np.flatnonzero(np.isin(folds, folds_scope))
    expected_ids = ids[expected_positions]
    by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        row_id = str(row["id"])
        if row_id in by_id:
            raise ValueError("candidate prediction IDs are duplicated")
        by_id[row_id] = row
    if set(by_id) != set(expected_ids):
        raise ValueError("candidate predictions do not exactly cover selected folds")
    result = bundle["qwen35_original_logit"].astype(np.float64).copy()
    for position in expected_positions:
        row = by_id[ids[position]]
        if int(row["fold"]) != int(folds[position]):
            raise ValueError("candidate prediction fold differs from replay")
        if str(row["category"]) != categories[position]:
            raise ValueError("candidate prediction category differs from replay")
        score = float(row["score"])
        if not np.isfinite(score):
            raise ValueError("candidate score is non-finite")
        if int(row["prediction"]) != int(score >= 0.0):
            raise ValueError("candidate prediction differs from frozen zero threshold")
        result[position] = score
    return result, hashes


def flammable_only_probabilities(
    *, bundle: dict[str, np.ndarray], candidate_logits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    categories = bundle["categories"].astype(str)
    baseline = LEGACY.sigmoid(bundle["qwen35_original_logit"])
    candidate = baseline.copy()
    flammable = categories == FLAMMABLE
    candidate[flammable] = LEGACY.sigmoid(candidate_logits[flammable])
    if not np.array_equal(candidate[~flammable], baseline[~flammable]):
        raise AssertionError("BAD Qwen route changed")
    return baseline, candidate


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    if positives == 0 or not np.isfinite(scores).all():
        raise ValueError("average precision requires finite scores and positive rows")
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    true_positives = np.cumsum(ranked_labels == 1)
    threshold_ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(scores) - 1]
    precision = true_positives[threshold_ends] / (threshold_ends + 1)
    recall = true_positives[threshold_ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def production_gates(metrics: dict[str, Any], *, stage: str) -> dict[str, bool]:
    ratio_gate = (
        metrics["corrected"] > 0
        if metrics["regressed"] == 0
        else metrics["corrected_to_regressed"] >= 1.5
    )
    common = {
        "bad_route_byte_identical": metrics["categories"]["БАД"]["delta"] == 0.0,
        "corrected_to_regressed_at_least_1_5": ratio_gate,
        "flammable_false_negatives_do_not_increase": (
            metrics["false_negatives"]["flammable"]["delta"] <= 0
        ),
        "all_positive_false_negatives_do_not_increase": (
            metrics["false_negatives"]["all_positive"]["delta"] <= 0
        ),
    }
    if stage == "screen":
        return {
            **common,
            "both_screen_folds_win": metrics["winning_folds"] == 2,
            "mean_fold_macro_delta_at_least_0_0015": metrics["mean_fold_delta"] >= 0.0015,
            "flammable_f1_does_not_drop": metrics["categories"][FLAMMABLE]["delta"] >= 0.0,
        }
    return {
        **common,
        "all_confirmation_folds_positive": all(
            metrics["folds"][str(fold)]["delta"] > 0 for fold in (1, 2, 4)
        ),
        "at_least_four_of_five_fold_wins": metrics["winning_folds"] >= 4,
        "macro_delta_at_least_0_006": metrics["macro_delta"] >= 0.006,
        "flammable_f1_delta_at_least_0_012": (
            metrics["categories"][FLAMMABLE]["delta"] >= 0.012
        ),
        "component_bootstrap_probability_at_least_0_90": (
            metrics["component_bootstrap"]["probability_delta_positive"] >= 0.90
        ),
    }


def evaluate(
    *,
    stage: str,
    bundle_path: Path,
    replay_contract_path: Path,
    registry_path: Path,
    candidate_paths: list[Path],
    output_path: Path,
    standalone_report_path: Path,
    screen_report_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite production-route evaluation")
    if stage not in {"screen", "full"}:
        raise ValueError("stage must be screen or full")
    expected_folds = SCREEN_FOLDS if stage == "screen" else FULL_FOLDS
    if len(candidate_paths) != len(expected_folds):
        raise ValueError("candidate path count differs from evaluation folds")
    if sha256_file(bundle_path) != FROZEN_BUNDLE_SHA256:
        raise ValueError("frozen production replay bundle checksum mismatch")
    if sha256_file(replay_contract_path) != FROZEN_CONTRACT_SHA256:
        raise ValueError("frozen production replay contract checksum mismatch")
    if sha256_file(registry_path) != FROZEN_REGISTRY_SHA256:
        raise ValueError("semantic-v3 registry checksum mismatch")
    standalone = json.loads(standalone_report_path.read_text(encoding="utf-8"))
    expected_decision = "OPEN_CONFIRMATION_FOLDS" if stage == "screen" else "ACCEPT_FOR_FULL_REFIT"
    if not (
        standalone.get("experiment_id") == "679"
        and standalone.get("mode") == stage
        and standalone.get("passed") is True
        and standalone.get("decision") == expected_decision
        and standalone.get("public_used") is False
        and standalone.get("sealed_rows") == 0
    ):
        raise ValueError("standalone 679 report does not authorize production evaluation")
    if stage == "full":
        if screen_report_path is None:
            raise ValueError("full production evaluation requires frozen production screen")
        screen = json.loads(screen_report_path.read_text(encoding="utf-8"))
        if not (
            screen.get("experiment_id") == "679"
            and screen.get("stage") == "screen"
            and screen.get("passed") is True
            and screen.get("decision") == "OPEN_CONFIRMATION"
        ):
            raise ValueError("production screen does not authorize full evaluation")
    elif screen_report_path is not None:
        raise ValueError("screen evaluation cannot consume a previous production screen")

    spec = LEGACY.load_spec()
    LEGACY.verify_source_recipe(spec)
    contract = LEGACY.verify_replay_contract(
        path=replay_contract_path,
        bundle_path=bundle_path,
        registry_path=registry_path,
        spec=spec,
    )
    registry = LEGACY._load_registry(registry_path, spec=spec)
    bundle = LEGACY.load_replay_bundle(path=bundle_path, contract=contract, registry=registry)
    candidate_logits, prediction_hashes = load_candidate_scores(
        candidate_paths, bundle=bundle, folds_scope=expected_folds
    )
    baseline_probability, candidate_probability = flammable_only_probabilities(
        bundle=bundle, candidate_logits=candidate_logits
    )
    baseline_prediction, baseline_score = LEGACY.route_predictions(
        bundle=bundle, qwen_probability=baseline_probability, route=spec["route"]
    )
    candidate_prediction, candidate_score = LEGACY.route_predictions(
        bundle=bundle, qwen_probability=candidate_probability, route=spec["route"]
    )
    labels = registry["label"].astype(np.int8).to_numpy()
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    components = bundle["semantic_components"].astype(str)
    metrics = LEGACY._metrics(
        labels=labels,
        categories=categories,
        folds=folds,
        components=components,
        baseline=baseline_prediction,
        candidate=candidate_prediction,
        selected_folds=expected_folds,
        bootstrap=(spec["bootstrap"] if stage == "full" else None),
    )
    selected_flammable = np.isin(folds, expected_folds) & (categories == FLAMMABLE)
    route_ap = {
        "baseline": average_precision(labels[selected_flammable], baseline_score[selected_flammable]),
        "candidate": average_precision(
            labels[selected_flammable], candidate_score[selected_flammable]
        ),
    }
    route_ap["delta"] = route_ap["candidate"] - route_ap["baseline"]
    gates = production_gates(metrics, stage=stage)
    passed = all(gates.values())
    decision = (
        "OPEN_CONFIRMATION"
        if stage == "screen" and passed
        else "REJECT_PRODUCTION_SCREEN"
        if stage == "screen"
        else "ACCEPT_FOR_REFIT"
        if passed
        else "REJECT_PRODUCTION_CANDIDATE"
    )
    result = {
        "schema_version": 1,
        "experiment_id": "679",
        "stage": stage,
        "changed_production_factor": "flammable_qwen35_adapter_only",
        "bad_route_byte_identical": True,
        "weights_changed": False,
        "thresholds_changed": False,
        "prior_rules_changed": False,
        "folds_evaluated": list(expected_folds),
        "metrics": metrics,
        "flammable_route_average_precision": route_ap,
        "gates": gates,
        "passed": passed,
        "decision": decision,
        "bundle_sha256": sha256_file(bundle_path),
        "replay_contract_sha256": sha256_file(replay_contract_path),
        "registry_sha256": sha256_file(registry_path),
        "standalone_report_sha256": sha256_file(standalone_report_path),
        "production_screen_sha256": (
            None if screen_report_path is None else sha256_file(screen_report_path)
        ),
        "candidate_prediction_sha256": prediction_hashes,
        "sealed_rows": 0,
        "public_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--stage", choices=("screen", "full"), required=True)
    result.add_argument("--bundle", type=Path, required=True)
    result.add_argument("--replay-contract", type=Path, required=True)
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--candidate-score", type=Path, action="append", required=True)
    result.add_argument("--standalone-report", type=Path, required=True)
    result.add_argument("--screen-report", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            evaluate(
                stage=args.stage,
                bundle_path=args.bundle,
                replay_contract_path=args.replay_contract,
                registry_path=args.registry,
                candidate_paths=args.candidate_score,
                output_path=args.output,
                standalone_report_path=args.standalone_report,
                screen_report_path=args.screen_report,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
