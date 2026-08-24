from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CATEGORIES = ("БАД", "Легковоспламеняющиеся")
SCREEN_FOLDS = (0, 3)
FULL_FOLDS = (0, 1, 2, 3, 4)
REQUIRED = {"global_index", "id", "fold", "category", "score", "prediction"}
FORBIDDEN = {"label", "target", "gold", "answer"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(((labels == 1) & predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    if positives == 0:
        raise ValueError("average precision requires a positive row")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    true_positives = np.cumsum(ranked_labels == 1)
    threshold_ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(ranked_scores) - 1]
    precision = true_positives[threshold_ends] / (threshold_ends + 1)
    recall = true_positives[threshold_ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def load_scores(paths: list[Path], expected_indices: list[int]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream)
    if any(not REQUIRED <= set(row) or FORBIDDEN & set(row) for row in records):
        raise ValueError("prediction schema or supervision mismatch")
    frame = pd.DataFrame([{key: row[key] for key in REQUIRED} for row in records])
    frame = frame.sort_values("global_index").reset_index(drop=True)
    if frame["global_index"].astype(int).tolist() != expected_indices:
        raise ValueError("predictions do not exactly cover the requested folds")
    if frame["id"].astype(str).duplicated().any() or not np.isfinite(frame["score"]).all():
        raise ValueError("duplicate IDs or non-finite scores")
    expected_predictions = frame["score"].to_numpy(np.float64) >= 0.0
    if not np.array_equal(frame["prediction"].astype(int), expected_predictions):
        raise ValueError("prediction differs from frozen zero threshold")
    return frame


def metric_block(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predictions = scores >= 0.0
    return {
        "rows": int(len(labels)),
        "positives": int((labels == 1).sum()),
        "f1": f1(labels, predictions),
        "average_precision": average_precision(labels, scores),
        "false_positives": int(((labels == 0) & predictions).sum()),
        "false_negatives": int(((labels == 1) & ~predictions).sum()),
    }


def delta_block(
    labels: np.ndarray, baseline_scores: np.ndarray, candidate_scores: np.ndarray
) -> dict[str, Any]:
    baseline = metric_block(labels, baseline_scores)
    candidate = metric_block(labels, candidate_scores)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "f1_delta": candidate["f1"] - baseline["f1"],
        "average_precision_delta": (
            candidate["average_precision"] - baseline["average_precision"]
        ),
    }


def evaluate(
    *,
    registry_path: Path,
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    mode: str,
    output_path: Path,
    screen_gate_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite an evaluation report")
    folds_scope = SCREEN_FOLDS if mode == "screen" else FULL_FOLDS
    if mode not in {"screen", "full"}:
        raise ValueError("mode must be screen or full")
    if len(baseline_paths) != len(folds_scope) or len(candidate_paths) != len(folds_scope):
        raise ValueError("prediction path count differs from fold scope")
    registry = pd.read_csv(registry_path, dtype={"id": str})
    registry = registry.loc[registry["split"].astype(str).eq("development")].copy()
    registry = registry.sort_values("id", key=lambda values: values.astype(str)).reset_index(
        drop=True
    )
    registry["global_index"] = np.arange(len(registry), dtype=np.int64)
    scope = registry.loc[registry["development_fold"].astype(int).isin(folds_scope)].copy()
    expected_indices = scope["global_index"].astype(int).tolist()
    baseline = load_scores(baseline_paths, expected_indices)
    candidate = load_scores(candidate_paths, expected_indices)
    scope = scope.sort_values("global_index").reset_index(drop=True)
    for frame in (baseline, candidate):
        if frame["id"].astype(str).tolist() != scope["id"].astype(str).tolist():
            raise ValueError("prediction IDs differ from immutable registry")
        if frame["fold"].astype(int).tolist() != scope["development_fold"].astype(int).tolist():
            raise ValueError("prediction folds differ from immutable registry")
        if frame["category"].astype(str).tolist() != scope["category"].astype(str).tolist():
            raise ValueError("prediction categories differ from immutable registry")

    labels = scope["label"].to_numpy(np.int8)
    folds = scope["development_fold"].to_numpy(np.int8)
    categories = scope["category"].astype(str).to_numpy()
    baseline_scores = baseline["score"].to_numpy(np.float64)
    candidate_scores = candidate["score"].to_numpy(np.float64)
    baseline_pred = baseline_scores >= 0.0
    candidate_pred = candidate_scores >= 0.0

    fold_metrics: dict[str, Any] = {}
    for fold in folds_scope:
        category_values: dict[str, Any] = {}
        for category in CATEGORIES:
            local = (folds == fold) & (categories == category)
            category_values[category] = delta_block(
                labels[local], baseline_scores[local], candidate_scores[local]
            )
        baseline_macro = float(
            np.mean([value["baseline"]["f1"] for value in category_values.values()])
        )
        candidate_macro = float(
            np.mean([value["candidate"]["f1"] for value in category_values.values()])
        )
        fold_metrics[str(fold)] = {
            "categories": category_values,
            "baseline_macro_f1": baseline_macro,
            "candidate_macro_f1": candidate_macro,
            "macro_f1_delta": candidate_macro - baseline_macro,
        }

    category_metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        local = categories == category
        category_metrics[category] = delta_block(
            labels[local], baseline_scores[local], candidate_scores[local]
        )
    corrected = int(((baseline_pred != labels) & (candidate_pred == labels)).sum())
    regressed = int(((baseline_pred == labels) & (candidate_pred != labels)).sum())
    ratio = None if regressed == 0 else corrected / regressed
    fold_deltas = {
        key: float(value["macro_f1_delta"]) for key, value in fold_metrics.items()
    }
    mean_macro_delta = float(np.mean(list(fold_deltas.values())))
    flammable_ap_deltas = {
        key: float(value["categories"]["Легковоспламеняющиеся"]["average_precision_delta"])
        for key, value in fold_metrics.items()
    }
    bad_ap_deltas = {
        key: float(value["categories"]["БАД"]["average_precision_delta"])
        for key, value in fold_metrics.items()
    }
    flammable_fn_delta = (
        category_metrics["Легковоспламеняющиеся"]["candidate"]["false_negatives"]
        - category_metrics["Легковоспламеняющиеся"]["baseline"]["false_negatives"]
    )
    ratio_gate = corrected > 0 if regressed == 0 else ratio is not None and ratio >= 1.5
    common_gates = {
        "no_pooled_category_f1_drop_below_minus_0_002": all(
            value["f1_delta"] >= -0.002 for value in category_metrics.values()
        ),
        "corrected_to_regressed_at_least_1_5": ratio_gate,
        "flammable_false_negatives_do_not_increase": flammable_fn_delta <= 0,
    }

    if mode == "screen":
        gates = {
            **common_gates,
            "both_screen_folds_macro_positive": all(delta > 0 for delta in fold_deltas.values()),
            "mean_macro_delta_at_least_0_0015": mean_macro_delta >= 0.0015,
            "both_screen_folds_flammable_ap_positive": all(
                delta > 0 for delta in flammable_ap_deltas.values()
            ),
            "mean_flammable_ap_delta_at_least_0_005": (
                float(np.mean(list(flammable_ap_deltas.values()))) >= 0.005
            ),
            "bad_ap_drop_at_most_0_002_each_fold": all(
                delta >= -0.002 for delta in bad_ap_deltas.values()
            ),
        }
        decision = "OPEN_CONFIRMATION_FOLDS" if all(gates.values()) else "REJECT_AT_SCREEN"
        screen_reproduced = None
        singleton = None
    else:
        if screen_gate_path is None:
            raise ValueError("full evaluation requires a frozen screen gate")
        screen_gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
        if screen_gate.get("decision") != "OPEN_CONFIRMATION_FOLDS" or not screen_gate.get(
            "passed"
        ):
            raise ValueError("screen gate is not accepted")
        screen_reproduced = all(
            abs(
                fold_metrics[str(fold)]["macro_f1_delta"]
                - screen_gate["folds"][str(fold)]["macro_f1_delta"]
            )
            <= 1e-12
            and abs(
                fold_metrics[str(fold)]["categories"]["Легковоспламеняющиеся"][
                    "average_precision_delta"
                ]
                - screen_gate["folds"][str(fold)]["categories"][
                    "Легковоспламеняющиеся"
                ]["average_precision_delta"]
            )
            <= 1e-12
            for fold in SCREEN_FOLDS
        )
        singleton_mask = scope["component_size"].astype(int).to_numpy() == 1
        singleton_category_f1 = []
        for category in CATEGORIES:
            local = singleton_mask & (categories == category)
            singleton_category_f1.append(
                f1(labels[local], candidate_pred[local]) - f1(labels[local], baseline_pred[local])
            )
        singleton = {
            "rows": int(singleton_mask.sum()),
            "macro_f1_delta": float(np.mean(singleton_category_f1)),
        }
        confirmation_positive = all(fold_deltas[str(fold)] > 0 for fold in (1, 2, 4))
        fold_wins = sum(delta > 0 for delta in fold_deltas.values())
        flammable_f1_delta = category_metrics["Легковоспламеняющиеся"]["f1_delta"]
        gates = {
            **common_gates,
            "screen_reproduced_exactly": screen_reproduced,
            "all_confirmation_folds_macro_positive": confirmation_positive,
            "at_least_four_of_five_macro_fold_wins": fold_wins >= 4,
            "mean_macro_delta_at_least_0_006": mean_macro_delta >= 0.006,
            "flammable_f1_delta_at_least_0_012": flammable_f1_delta >= 0.012,
            "at_least_four_of_five_flammable_ap_fold_wins": (
                sum(delta > 0 for delta in flammable_ap_deltas.values()) >= 4
            ),
            "mean_flammable_ap_delta_at_least_0_005": (
                float(np.mean(list(flammable_ap_deltas.values()))) >= 0.005
            ),
            "bad_ap_drop_at_most_0_002_each_fold": all(
                delta >= -0.002 for delta in bad_ap_deltas.values()
            ),
            "semantic_singleton_macro_positive": singleton["macro_f1_delta"] > 0,
        }
        decision = "ACCEPT_FOR_FULL_REFIT" if all(gates.values()) else "REJECT_FULL_CANDIDATE"

    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "679",
        "control_experiment_id": "641",
        "mode": mode,
        "evaluation_version": "semantic_family_v3",
        "changed_factor": "learning_rate_only_0.0002_to_0.0001",
        "folds": fold_metrics,
        "mean_macro_f1_delta": mean_macro_delta,
        "mean_flammable_average_precision_delta": float(
            np.mean(list(flammable_ap_deltas.values()))
        ),
        "categories": category_metrics,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "flammable_false_negative_delta": flammable_fn_delta,
        "semantic_singleton": singleton,
        "screen_reproduced": screen_reproduced,
        "threshold": 0.0,
        "threshold_tuned": False,
        "prediction_sha256": {
            "control": [sha256_file(path) for path in baseline_paths],
            "candidate": [sha256_file(path) for path in candidate_paths],
        },
        "screen_gate_sha256": (
            sha256_file(screen_gate_path) if screen_gate_path is not None else None
        ),
        "gates": gates,
        "passed": passed,
        "sealed_rows": 0,
        "public_used": False,
        "decision": decision,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--baseline-score", type=Path, action="append", required=True)
    result.add_argument("--candidate-score", type=Path, action="append", required=True)
    result.add_argument("--mode", choices=("screen", "full"), required=True)
    result.add_argument("--screen-gate", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            evaluate(
                registry_path=args.registry,
                baseline_paths=args.baseline_score,
                candidate_paths=args.candidate_score,
                mode=args.mode,
                output_path=args.output,
                screen_gate_path=args.screen_gate,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
