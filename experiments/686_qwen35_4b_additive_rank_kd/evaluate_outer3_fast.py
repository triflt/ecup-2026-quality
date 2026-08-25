from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import evaluate as frozen
import numpy as np
import train_pair_fold as trainer
from build_pair_runtime import canonical_sha256

EXPERIMENT_ID = "686"
FLAMMABLE = trainer.FLAMMABLE
OUTER_FOLD = 3
EXPECTED_REGISTRY_SHA256 = (
    "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae"
)
EXPECTED_DEVELOPMENT_ROWS = 11118
EXPECTED_OUTER3_FLAMMABLE_ROWS = 943
REGISTRY_COLUMNS = (
    "id",
    "category",
    "label",
    "semantic_component",
    "component_size",
    "split",
    "development_fold",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_prediction_rows(
    path: Path, expected: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    rows = read_jsonl(path)
    if len(rows) != len(expected):
        raise ValueError("prediction row count differs from outer3 validation")
    forbidden = {"label", "target", "gold", "sealed", "public"}
    scores: list[float] = []
    predictions: list[int] = []
    for prediction, runtime_row in zip(rows, expected, strict=True):
        if set(prediction) & forbidden:
            raise ValueError("prediction payload contains supervision")
        for field in ("global_index", "id", "fold", "category"):
            if prediction.get(field) != runtime_row.get(field):
                raise ValueError("prediction/runtime binding mismatch")
        score = float(prediction["score"])
        verdict = int(prediction["prediction"])
        if (
            not math.isfinite(score)
            or verdict not in (0, 1)
            or verdict != int(score >= 0.0)
        ):
            raise ValueError("prediction score or frozen threshold mismatch")
        scores.append(score)
        predictions.append(verdict)
    return np.asarray(scores, dtype=np.float64), np.asarray(predictions, dtype=np.int8)


def bind_registry_rows(
    validation: list[dict[str, Any]],
    development: list[dict[str, Any]],
    *,
    expected_selected_rows: int,
) -> tuple[np.ndarray, list[str], str]:
    if len(development) != len({str(row["id"]) for row in development}):
        raise ValueError("duplicate development ID in semantic registry")
    ordered = sorted(development, key=lambda row: str(row["id"]))
    selected: list[dict[str, Any]] = []
    for global_index, row in enumerate(ordered):
        fold = int(row["development_fold"])
        label = int(row["label"])
        if fold not in range(5) or label not in (0, 1):
            raise ValueError("invalid fold or label in semantic registry")
        if fold == OUTER_FOLD and str(row["category"]) == FLAMMABLE:
            selected.append(
                {
                    "global_index": global_index,
                    "id": str(row["id"]),
                    "fold": fold,
                    "category": str(row["category"]),
                    "label": label,
                    "semantic_component": str(row["semantic_component"]),
                }
            )
    if len(selected) != expected_selected_rows:
        raise ValueError("outer3 flammable registry coverage mismatch")
    runtime_keys = [
        (
            int(row["global_index"]),
            str(row["id"]),
            int(row["fold"]),
            str(row["category"]),
        )
        for row in validation
    ]
    registry_keys = [
        (
            int(row["global_index"]),
            str(row["id"]),
            int(row["fold"]),
            str(row["category"]),
        )
        for row in selected
    ]
    if runtime_keys != registry_keys:
        raise ValueError("outer3 validation differs from frozen registry packet")
    labels: list[int] = []
    components: list[str] = []
    packet: list[dict[str, Any]] = []
    for row in selected:
        labels.append(int(row["label"]))
        components.append(str(row["semantic_component"]))
        packet.append(dict(row))
    return (
        np.asarray(labels, dtype=np.int8),
        components,
        canonical_sha256(packet),
    )


def bind_outer3_labels(
    validation: list[dict[str, Any]], registry_path: Path
) -> tuple[np.ndarray, list[str], str]:
    if frozen.sha256_file(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("semantic-v3 registry checksum mismatch")
    with registry_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REGISTRY_COLUMNS:
            raise ValueError("semantic-v3 registry schema mismatch")
        rows = list(reader)
    development = [row for row in rows if row["split"] == "development"]
    if len(development) != EXPECTED_DEVELOPMENT_ROWS:
        raise ValueError("semantic-v3 development row count mismatch")
    return bind_registry_rows(
        validation,
        development,
        expected_selected_rows=EXPECTED_OUTER3_FLAMMABLE_ROWS,
    )


def positive_class_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def correction_ratio(corrected: int, regressed: int) -> float | None:
    return None if regressed == 0 else corrected / regressed


def correction_ratio_ok(corrected: int, regressed: int, minimum: float) -> bool:
    return corrected > 0 if regressed == 0 else corrected / regressed >= minimum


def score_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    def describe(values: np.ndarray) -> dict[str, float]:
        if not len(values):
            raise ValueError("score diagnostic cohort is empty")
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q50": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
        }

    return {
        "all": describe(scores),
        "label0": describe(scores[labels == 0]),
        "label1": describe(scores[labels == 1]),
        "within_abs_0_25": int(np.sum(np.abs(scores) <= 0.25)),
        "within_abs_0_50": int(np.sum(np.abs(scores) <= 0.50)),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.technical_smoke:
        raise ValueError("fast evaluator is technical-only; use frozen full replay")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite fast evaluation")
    control_paths, control_acceptances = frozen.load_acceptances(
        [args.control_score],
        [args.control_acceptance],
        folds_scope=(OUTER_FOLD,),
        mode="paired_hard_control",
        technical_smoke=args.technical_smoke,
    )
    candidate_paths, candidate_acceptances = frozen.load_acceptances(
        [args.candidate_score],
        [args.candidate_acceptance],
        folds_scope=(OUTER_FOLD,),
        mode="rank_candidate",
        technical_smoke=args.technical_smoke,
    )
    frozen.verify_paired_contracts(control_acceptances, candidate_acceptances)
    frozen.verify_promotion_lineage(
        control_acceptances,
        stage="blind3",
        promotion=None,
        promotion_file_sha256=None,
        promotion_evaluation_file_sha256=None,
        promotion_source=None,
    )
    frozen.verify_promotion_lineage(
        candidate_acceptances,
        stage="blind3",
        promotion=None,
        promotion_file_sha256=None,
        promotion_evaluation_file_sha256=None,
        promotion_source=None,
    )
    code_acceptance = frozen.load_evaluation_code_acceptance(args.code_acceptance)

    _, source3_validation, source3_audit = trainer.control.load_runtime(
        args.source_fold3_runtime, trainer.SOURCE_EXPERIMENT_ID, OUTER_FOLD
    )
    if source3_audit["contract_sha256"] != args.expected_source_fold3_contract:
        raise ValueError("fold3 source-runtime contract mismatch")
    if (
        source3_audit["contract_sha256"]
        != control_acceptances[OUTER_FOLD]["source_641_runtime_contract_sha256"]
    ):
        raise ValueError("training artifact and fold3 source runtime differ")

    full_validation = [
        row for row in source3_validation if row["category"] == FLAMMABLE
    ]
    labels, components, label_packet_sha256 = bind_outer3_labels(
        full_validation, args.registry
    )
    if args.technical_smoke:
        validation = full_validation[:2]
        labels = labels[:2]
        components = components[:2]
    else:
        validation = full_validation
    if len(validation) != int(control_acceptances[OUTER_FOLD]["rows"]):
        raise ValueError("accepted prediction count differs from flammable validation")
    control_scores, control_predictions = load_prediction_rows(
        control_paths[0], validation
    )
    candidate_scores, candidate_predictions = load_prediction_rows(
        candidate_paths[0], validation
    )

    if args.technical_smoke:
        result = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "stage": "outer3_blind_fast_remote_technical_smoke",
            "folds": [OUTER_FOLD],
            "label_source": "frozen_semantic_v3_registry_exact_ordered_packet",
            "changed_factor": "none_technical_transport_and_binding_only",
            "rows": len(labels),
            "candidate_prediction_sha256": frozen.sha256_file(candidate_paths[0]),
            "control_prediction_sha256": frozen.sha256_file(control_paths[0]),
            "candidate_acceptance_sha256": candidate_acceptances[OUTER_FOLD][
                "acceptance_sha256"
            ],
            "control_acceptance_sha256": control_acceptances[OUTER_FOLD][
                "acceptance_sha256"
            ],
            "evaluation_code_bundle_sha256": code_acceptance["bundle_sha256"],
            "evaluation_code_revision": code_acceptance["git_revision"],
            "evaluation_code_acceptance_sha256": code_acceptance[
                "acceptance_sha256"
            ],
            "source_fold3_contract_sha256": source3_audit["contract_sha256"],
            "registry_sha256": EXPECTED_REGISTRY_SHA256,
            "label_packet_rows": len(full_validation),
            "label_packet_sha256": label_packet_sha256,
            "label_packet_full_coverage_verified": True,
            "candidate_scores_finite": bool(np.isfinite(candidate_scores).all()),
            "control_scores_finite": bool(np.isfinite(control_scores).all()),
            "validation_labels_read_by_training": 0,
            "validation_labels_read_by_evaluator": len(full_validation),
            "sealed_rows": 0,
            "public_used": False,
            "threshold_tuned": False,
            "decision": "ACCEPT_FAST_EVAL_TECHNICAL_SMOKE",
            "authoritative_scope": "technical_only_no_scientific_promotion",
        }
        result["evaluation_sha256"] = canonical_sha256(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    control_metrics = positive_class_metrics(labels, control_predictions)
    candidate_metrics = positive_class_metrics(labels, candidate_predictions)
    ap_control = frozen.BASE.average_precision(labels, control_scores)
    ap_candidate = frozen.BASE.average_precision(labels, candidate_scores)
    corrected_mask = (control_predictions != labels) & (candidate_predictions == labels)
    regressed_mask = (control_predictions == labels) & (candidate_predictions != labels)
    corrected = int(corrected_mask.sum())
    regressed = int(regressed_mask.sum())
    score_delta = candidate_scores - control_scores
    direct_flammable_f1_delta = candidate_metrics["f1"] - control_metrics["f1"]
    direct_macro_delta = direct_flammable_f1_delta / 2.0
    component_corrected = len(
        {component for component, flag in zip(components, corrected_mask, strict=True) if flag}
    )
    component_regressed = len(
        {component for component, flag in zip(components, regressed_mask, strict=True) if flag}
    )
    gates = {
        "bad_route_byte_identical_by_frozen_flammable_only_route": True,
        "flammable_false_negatives_do_not_increase": (
            candidate_metrics["fn"] <= control_metrics["fn"]
        ),
        "outer3_ap_positive": ap_candidate > ap_control,
        "outer3_direct_macro_positive": direct_macro_delta > 0.0,
        "outer3_corrections_to_regressions_at_least_1_2": correction_ratio_ok(
            corrected, regressed, 1.2
        ),
    }
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "outer3_blind_fast_remote",
        "folds": [OUTER_FOLD],
        "label_source": "frozen_semantic_v3_registry_exact_ordered_packet",
        "changed_factor": "add_rank_loss_0.5_without_reducing_hard_bce",
        "flammable_average_precision": {
            "control": ap_control,
            "candidate": ap_candidate,
            "delta": ap_candidate - ap_control,
        },
        "flammable_metrics": {
            "control": control_metrics,
            "candidate": candidate_metrics,
            "f1_delta": direct_flammable_f1_delta,
            "fn_delta": candidate_metrics["fn"] - control_metrics["fn"],
        },
        "direct_macro_delta_with_bad_byte_identical": direct_macro_delta,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": correction_ratio(corrected, regressed),
        "corrected_components": component_corrected,
        "regressed_components": component_regressed,
        "score_diagnostics": {
            "control": score_summary(labels, control_scores),
            "candidate": score_summary(labels, candidate_scores),
            "candidate_minus_control": score_summary(labels, score_delta),
            "flips_0_to_1": int(
                np.sum((control_predictions == 0) & (candidate_predictions == 1))
            ),
            "flips_1_to_0": int(
                np.sum((control_predictions == 1) & (candidate_predictions == 0))
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "OPEN_CONFIRMATION_FOLDS124" if all(gates.values()) else "REJECT_AT_OUTER3",
        "candidate_prediction_sha256": frozen.sha256_file(candidate_paths[0]),
        "control_prediction_sha256": frozen.sha256_file(control_paths[0]),
        "candidate_acceptance_sha256": candidate_acceptances[OUTER_FOLD][
            "acceptance_sha256"
        ],
        "control_acceptance_sha256": control_acceptances[OUTER_FOLD][
            "acceptance_sha256"
        ],
        "evaluation_code_bundle_sha256": code_acceptance["bundle_sha256"],
        "evaluation_code_revision": code_acceptance["git_revision"],
        "evaluation_code_acceptance_sha256": code_acceptance[
            "acceptance_sha256"
        ],
        "source_fold3_contract_sha256": source3_audit["contract_sha256"],
        "registry_sha256": EXPECTED_REGISTRY_SHA256,
        "label_packet_rows": len(full_validation),
        "label_packet_sha256": label_packet_sha256,
        "label_packet_full_coverage_verified": True,
        "validation_rows": len(validation),
        "validation_labels_read_by_training": 0,
        "validation_labels_read_by_evaluator": len(labels),
        "sealed_rows": 0,
        "public_used": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "authoritative_scope": "outer3_blind_promotion_only",
    }
    result["evaluation_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-score", type=Path, required=True)
    parser.add_argument("--candidate-acceptance", type=Path, required=True)
    parser.add_argument("--control-score", type=Path, required=True)
    parser.add_argument("--control-acceptance", type=Path, required=True)
    parser.add_argument("--source-fold3-runtime", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-source-fold3-contract", required=True)
    parser.add_argument("--code-acceptance", type=Path, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(evaluate(arguments), ensure_ascii=False, indent=2, sort_keys=True))
