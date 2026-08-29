from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "677"
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
REQUESTED_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def average_precision(labels: Iterable[int], scores: Iterable[float]) -> float:
    groups: dict[float, list[int]] = {}
    positives = count = 0
    for raw_label, raw_score in zip(labels, scores, strict=True):
        label, score = int(raw_label), float(raw_score)
        if label not in {0, 1} or not math.isfinite(score):
            raise ValueError("AP expects binary labels and finite scores")
        group = groups.setdefault(score, [0, 0])
        group[0] += label
        group[1] += 1
        positives += label
        count += 1
    if count == 0 or positives == 0:
        raise ValueError("AP requires rows and at least one positive")
    tp = predicted = 0
    result = 0.0
    for score in sorted(groups, reverse=True):
        group_positives, group_count = groups[score]
        tp += group_positives
        predicted += group_count
        result += (group_positives / positives) * (tp / predicted)
    return result


def load_registry(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        item_id = int(row["id"])
        if item_id in result:
            raise ValueError(f"duplicate registry id: {item_id}")
        result[item_id] = {
            "category": row["category"],
            "label": int(row["label"]),
            "fold": int(row["development_fold"]) if row["development_fold"] else None,
            "split": row["split"],
        }
    return result


def read_predictions(path: Path) -> dict[int, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            item_id = int(row["id"])
            if item_id in result:
                raise ValueError(f"duplicate prediction id in {path}: {item_id}")
            result[item_id] = row
    return result


def score_checkpoint(
    *, registry: dict[int, dict[str, Any]], path: Path, inner_fold: int
) -> dict[str, Any]:
    predictions = read_predictions(path)
    expected = {
        item_id
        for item_id, row in registry.items()
        if row["split"] == "development" and row["fold"] == inner_fold
    }
    if set(predictions) != expected:
        raise ValueError(f"prediction coverage mismatch for inner fold {inner_fold}")
    metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        ids = sorted(item_id for item_id in expected if registry[item_id]["category"] == category)
        labels = [registry[item_id]["label"] for item_id in ids]
        scores = [float(predictions[item_id]["score"]) for item_id in ids]
        metrics[category] = {
            "rows": len(ids),
            "positives": sum(labels),
            "average_precision": average_precision(labels, scores),
        }
    return metrics


def evaluate(registry_path: Path, run_dirs: list[Path], output: Path) -> dict[str, Any]:
    if len(run_dirs) != 4:
        raise ValueError("exactly four inner runs are required")
    registry = load_registry(registry_path)
    fold_results = []
    for expected_fold, run_dir in zip((1, 2, 3, 4), run_dirs, strict=True):
        report_path = run_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        payload = dict(report)
        digest = payload.pop("contract_sha256", None)
        expected = {
            "experiment_id": EXPERIMENT_ID,
            "outer_screen_fold": 0,
            "inner_validation_fold": expected_fold,
            "blind_confirmation_folds": [0],
            "fold3_is_blind": False,
            "changed_factor": "optimizer_stop_fraction_only",
            "technical_smoke": False,
            "validation_labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "threshold_tuned": False,
            "decision": "READY_FOR_INNER_EVALUATION",
        }
        if digest != canonical_sha256(payload) or any(
            report.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"run report contract mismatch for fold {expected_fold}")
        if tuple(report["checkpoint_fractions"]) != REQUESTED_FRACTIONS:
            raise ValueError("checkpoint fraction contract drift")
        checkpoints = []
        for checkpoint in report["checkpoints"]:
            prediction_path = run_dir / checkpoint["predictions"]
            if checkpoint["predictions_sha256"] != sha256_file(prediction_path):
                raise ValueError("prediction checksum mismatch")
            requested_fraction = float(checkpoint["requested_training_fraction"])
            checkpoints.append(
                {
                    "requested_training_fraction": requested_fraction,
                    "actual_training_fraction": float(checkpoint["training_fraction"]),
                    "optimizer_step": int(checkpoint["optimizer_step"]),
                    "prediction_sha256": checkpoint["predictions_sha256"],
                    "metrics": score_checkpoint(
                        registry=registry, path=prediction_path, inner_fold=expected_fold
                    ),
                }
            )
        fold_results.append(
            {
                "fold": expected_fold,
                "report_sha256": sha256_file(report_path),
                "checkpoints": checkpoints,
            }
        )
    baseline_fraction = 1.0
    candidates = []
    for fraction in REQUESTED_FRACTIONS[:-1]:
        per_fold = []
        for fold in fold_results:
            by_fraction = {
                checkpoint["requested_training_fraction"]: checkpoint
                for checkpoint in fold["checkpoints"]
            }
            candidate, baseline = by_fraction[fraction], by_fraction[baseline_fraction]
            flammable_delta = (
                candidate["metrics"]["Легковоспламеняющиеся"]["average_precision"]
                - baseline["metrics"]["Легковоспламеняющиеся"]["average_precision"]
            )
            bad_delta = (
                candidate["metrics"]["БАД"]["average_precision"]
                - baseline["metrics"]["БАД"]["average_precision"]
            )
            per_fold.append(
                {"fold": fold["fold"], "flammable_ap_delta": flammable_delta, "bad_ap_delta": bad_delta}
            )
        mean_flammable = sum(row["flammable_ap_delta"] for row in per_fold) / len(per_fold)
        mean_bad = sum(row["bad_ap_delta"] for row in per_fold) / len(per_fold)
        wins = sum(row["flammable_ap_delta"] > 0 for row in per_fold)
        worst_bad = min(row["bad_ap_delta"] for row in per_fold)
        accepted = wins >= 3 and mean_flammable >= 0.01 and worst_bad >= -0.002
        candidates.append(
            {
                "requested_training_fraction": fraction,
                "per_fold": per_fold,
                "flammable_fold_wins": wins,
                "mean_flammable_ap_delta": mean_flammable,
                "mean_bad_ap_delta": mean_bad,
                "worst_bad_ap_delta": worst_bad,
                "accepted": accepted,
            }
        )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    selected = (
        max(accepted, key=lambda row: (row["mean_flammable_ap_delta"], -row["requested_training_fraction"]))
        if accepted
        else None
    )
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "metric_definition": {
            "name": "average_precision",
            "equivalent": "sklearn.metrics.average_precision_score",
            "ties": "equal scores form one threshold group",
        },
        "registry_sha256": sha256_file(registry_path),
        "outer_screen_fold": 0,
        "inner_folds": [1, 2, 3, 4],
        "selection_scope": {
            "blind_confirmation_folds": [0],
            "fold3_is_blind": False,
            "fold3_reason": "fold 3 participates in donor-inner stop-fraction selection",
            "required_follow_up": (
                "A separate nested selector built only inside outer-fold-3 training data is required "
                "before evaluating outer fold 3."
            ),
        },
        "fold_results": fold_results,
        "candidate_deltas_vs_final": candidates,
        "selected_training_fraction": (
            selected["requested_training_fraction"] if selected is not None else None
        ),
        "sealed_rows": 0,
        "public_used": False,
        "threshold_tuned": False,
        "decision": (
            "GO_CONFIRM_SELECTED_STOP_ON_OUTER_FOLD_0_ONLY"
            if selected is not None
            else "NO_GO_KEEP_FINAL_CHECKPOINT_RECIPE"
        ),
    }
    result["contract_sha256"] = canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--run-dir", type=Path, action="append", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(
        json.dumps(
            evaluate(arguments.registry, arguments.run_dir, arguments.output),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
