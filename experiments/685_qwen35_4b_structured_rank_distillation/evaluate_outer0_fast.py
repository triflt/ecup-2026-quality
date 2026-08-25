from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import evaluate as frozen
import numpy as np
import train_pair_fold as trainer
from build_pair_runtime import canonical_sha256

EXPERIMENT_ID = "685"
FLAMMABLE = trainer.FLAMMABLE
OUTER_FOLD = 0
LABEL_DONOR_FOLD = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_prediction_rows(
    path: Path, expected: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    rows = read_jsonl(path)
    if len(rows) != len(expected):
        raise ValueError("prediction row count differs from outer0 validation")
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


def bind_outer0_labels(
    validation: list[dict[str, Any]], donor_train: list[dict[str, Any]]
) -> tuple[np.ndarray, list[str]]:
    donor_by_index: dict[int, dict[str, Any]] = {}
    for row in donor_train:
        global_index = int(row["global_index"])
        if global_index in donor_by_index:
            existing = donor_by_index[global_index]
            identity = ("global_index", "id", "fold", "category", "label", "semantic_component")
            if any(existing.get(field) != row.get(field) for field in identity):
                raise ValueError("conflicting duplicate in label-donor train runtime")
            continue
        donor_by_index[global_index] = row
    labels: list[int] = []
    components: list[str] = []
    for row in validation:
        donor = donor_by_index.get(int(row["global_index"]))
        if donor is None:
            raise ValueError("outer0 validation row is absent from fold3 train donor")
        for field in ("global_index", "id", "fold", "category"):
            if donor.get(field) != row.get(field):
                raise ValueError("label donor/runtime binding mismatch")
        label = int(donor["label"])
        if label not in (0, 1):
            raise ValueError("invalid donor label")
        labels.append(label)
        components.append(str(donor["semantic_component"]))
    return np.asarray(labels, dtype=np.int8), components


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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
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

    _, source0_validation, source0_audit = trainer.control.load_runtime(
        args.source_fold0_runtime, trainer.SOURCE_EXPERIMENT_ID, OUTER_FOLD
    )
    donor_train, _, donor_audit = trainer.control.load_runtime(
        args.label_donor_runtime, trainer.SOURCE_EXPERIMENT_ID, LABEL_DONOR_FOLD
    )
    if source0_audit["contract_sha256"] != args.expected_source_fold0_contract:
        raise ValueError("fold0 source-runtime contract mismatch")
    if donor_audit["contract_sha256"] != args.expected_label_donor_contract:
        raise ValueError("label-donor source-runtime contract mismatch")
    if (
        source0_audit["contract_sha256"]
        != control_acceptances[OUTER_FOLD]["source_641_runtime_contract_sha256"]
    ):
        raise ValueError("training artifact and fold0 source runtime differ")

    validation = [
        row for row in source0_validation if row["category"] == FLAMMABLE
    ]
    if args.technical_smoke:
        validation = validation[:2]
    if len(validation) != int(control_acceptances[OUTER_FOLD]["rows"]):
        raise ValueError("accepted prediction count differs from flammable validation")
    labels, components = bind_outer0_labels(validation, donor_train)
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
            "stage": "outer0_fast_remote_technical_smoke",
            "folds": [OUTER_FOLD],
            "label_source": "fold3_outer_train_runtime_exact_global_index_join",
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
            "source_fold0_contract_sha256": source0_audit["contract_sha256"],
            "label_donor_contract_sha256": donor_audit["contract_sha256"],
            "candidate_scores_finite": bool(np.isfinite(candidate_scores).all()),
            "control_scores_finite": bool(np.isfinite(control_scores).all()),
            "validation_labels_read_by_training": 0,
            "validation_labels_read_by_evaluator": len(labels),
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
        "outer0_ap_positive": ap_candidate > ap_control,
        "outer0_direct_macro_nonnegative": direct_macro_delta >= 0.0,
        "outer0_corrections_to_regressions_at_least_1_2": correction_ratio_ok(
            corrected, regressed, 1.2
        ),
    }
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "outer0_fast_remote",
        "folds": [OUTER_FOLD],
        "label_source": "fold3_outer_train_runtime_exact_global_index_join",
        "changed_factor": "add_fixed_rank_loss_weight_0.5_to_paired_hard_control",
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
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "OPEN_SCREEN_FOLD3" if all(gates.values()) else "REJECT_AT_OUTER0",
        "candidate_prediction_sha256": frozen.sha256_file(candidate_paths[0]),
        "control_prediction_sha256": frozen.sha256_file(control_paths[0]),
        "candidate_acceptance_sha256": candidate_acceptances[OUTER_FOLD][
            "acceptance_sha256"
        ],
        "control_acceptance_sha256": control_acceptances[OUTER_FOLD][
            "acceptance_sha256"
        ],
        "source_fold0_contract_sha256": source0_audit["contract_sha256"],
        "label_donor_contract_sha256": donor_audit["contract_sha256"],
        "validation_rows": len(validation),
        "validation_labels_read_by_training": 0,
        "validation_labels_read_by_evaluator": len(labels),
        "sealed_rows": 0,
        "public_used": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "authoritative_scope": "outer0_promotion_only",
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
    parser.add_argument("--source-fold0-runtime", type=Path, required=True)
    parser.add_argument("--label-donor-runtime", type=Path, required=True)
    parser.add_argument("--expected-source-fold0-contract", required=True)
    parser.add_argument("--expected-label-donor-contract", required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(evaluate(arguments), ensure_ascii=False, indent=2, sort_keys=True))
