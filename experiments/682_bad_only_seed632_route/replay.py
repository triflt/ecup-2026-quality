from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from contract import add_self_hash, load_spec, require_sha, sha256_file

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"


def _load_evaluator(path: Path, expected_sha256: str):
    require_sha(path, expected_sha256, name="experiment-635 evaluator")
    spec = importlib.util.spec_from_file_location("_exp682_frozen_exp635_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def category_changes(
    *, labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, categories: np.ndarray
) -> dict[str, dict[str, int | float | None]]:
    result: dict[str, dict[str, int | float | None]] = {}
    for category in (BAD, FLAMMABLE):
        mask = categories == category
        corrected = int((mask & (baseline != labels) & (candidate == labels)).sum())
        regressed = int((mask & (baseline == labels) & (candidate != labels)).sum())
        result[category] = {
            "corrected": corrected,
            "regressed": regressed,
            "corrected_to_regressed": None if regressed == 0 else corrected / regressed,
        }
    return result


def fold_changes(
    *,
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for fold in range(5):
        mask = folds == fold
        corrected = int((mask & (baseline != labels) & (candidate == labels)).sum())
        regressed = int((mask & (baseline == labels) & (candidate != labels)).sum())
        result[str(fold)] = {
            "corrected": corrected,
            "regressed": regressed,
            "corrected_to_regressed": None if regressed == 0 else corrected / regressed,
            "categories": category_changes(
                labels=labels[mask],
                baseline=baseline[mask],
                candidate=candidate[mask],
                categories=categories[mask],
            ),
        }
    return result


def _bytes_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    frozen = load_spec(args.spec)
    inputs = frozen["frozen_inputs"]
    require_sha(args.bundle, inputs["replay_bundle_sha256"], name="replay bundle")
    require_sha(
        args.replay_contract,
        inputs["replay_contract_file_sha256"],
        name="replay contract",
    )
    require_sha(
        args.accepted_632_report,
        inputs["accepted_632_report_sha256"],
        name="accepted experiment-632 report",
    )
    require_sha(args.registry, inputs["semantic_v3_registry_sha256"], name="registry")
    evaluator = _load_evaluator(args.evaluator, inputs["exp635_evaluator_sha256"])
    require_sha(
        args.evaluator.with_name("frozen_spec.json"),
        inputs["exp635_frozen_spec_sha256"],
        name="experiment-635 frozen spec",
    )

    upstream_spec = evaluator.load_spec()
    evaluator.verify_source_recipe(upstream_spec)
    contract = evaluator.verify_replay_contract(
        path=args.replay_contract,
        bundle_path=args.bundle,
        registry_path=args.registry,
        spec=upstream_spec,
    )
    if contract.get("contract_sha256") != inputs["replay_contract_canonical_sha256"]:
        raise ValueError("replay contract canonical checksum mismatch")
    if contract.get("route") != frozen["route"]:
        raise ValueError("experiment-682 route differs from the frozen system-140 contract")
    evaluator.verify_accepted_632(args.accepted_632_report, contract)
    registry = evaluator._load_registry(args.registry, spec=upstream_spec)
    bundle = evaluator.load_replay_bundle(
        path=args.bundle,
        contract=contract,
        registry=registry,
    )

    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    labels = registry["label"].to_numpy(np.int8)
    components = bundle["semantic_components"].astype(str)
    original_logits = bundle["qwen35_original_logit"]
    seed632_logits = bundle["qwen35_seed632_logit"]
    routed_logits = original_logits.copy()
    bad_mask = categories == BAD
    flammable_mask = categories == FLAMMABLE
    routed_logits[bad_mask] = seed632_logits[bad_mask]
    if not np.array_equal(routed_logits[flammable_mask], original_logits[flammable_mask]):
        raise AssertionError("flammable raw logits changed")

    baseline_probability = evaluator.sigmoid(original_logits)
    candidate_probability = evaluator.sigmoid(routed_logits)
    baseline_prediction, baseline_score = evaluator.route_predictions(
        bundle=bundle,
        qwen_probability=baseline_probability,
        route=frozen["route"],
    )
    candidate_prediction, candidate_score = evaluator.route_predictions(
        bundle=bundle,
        qwen_probability=candidate_probability,
        route=frozen["route"],
    )
    identity_checks = {
        "route_config_equal": True,
        "robust_base_array_reused": True,
        "qwen3vl_array_reused": True,
        "prior_override_array_reused": True,
        "flammable_logit_array_equal": bool(
            np.array_equal(routed_logits[flammable_mask], original_logits[flammable_mask])
        ),
        "flammable_probability_array_equal": bool(
            np.array_equal(
                candidate_probability[flammable_mask],
                baseline_probability[flammable_mask],
            )
        ),
        "flammable_final_score_array_equal": bool(
            np.array_equal(candidate_score[flammable_mask], baseline_score[flammable_mask])
        ),
        "flammable_final_prediction_array_equal": bool(
            np.array_equal(
                candidate_prediction[flammable_mask],
                baseline_prediction[flammable_mask],
            )
        ),
        "flammable_original_logit_bytes_sha256": _bytes_sha256(
            original_logits[flammable_mask]
        ),
        "flammable_routed_logit_bytes_sha256": _bytes_sha256(routed_logits[flammable_mask]),
    }
    if not all(value for key, value in identity_checks.items() if key.endswith("array_equal")):
        raise AssertionError("flammable route is not byte-identical")

    metrics = evaluator._metrics(
        labels=labels,
        categories=categories,
        folds=folds,
        components=components,
        baseline=baseline_prediction,
        candidate=candidate_prediction,
        selected_folds=[0, 1, 2, 3, 4],
        bootstrap=frozen["bootstrap"],
    )
    metrics["changes_by_category"] = category_changes(
        labels=labels,
        baseline=baseline_prediction,
        candidate=candidate_prediction,
        categories=categories,
    )
    metrics["changes_by_fold"] = fold_changes(
        labels=labels,
        baseline=baseline_prediction,
        candidate=candidate_prediction,
        categories=categories,
        folds=folds,
    )
    metrics["category_rows"] = {
        category: int((categories == category).sum()) for category in (BAD, FLAMMABLE)
    }
    metrics["macro_delta_contribution"] = {
        category: metrics["categories"][category]["delta"] / 2.0
        for category in (BAD, FLAMMABLE)
    }
    metrics["false_negatives_by_category"] = {}
    for category in (BAD, FLAMMABLE):
        mask = (categories == category) & (labels == 1)
        before = int((mask & (baseline_prediction == 0)).sum())
        after = int((mask & (candidate_prediction == 0)).sum())
        metrics["false_negatives_by_category"][category] = {
            "baseline": before,
            "candidate": after,
            "delta": after - before,
        }
    gate = frozen["acceptance"]
    gates = {
        "macro_delta": metrics["macro_delta"] >= gate["minimum_macro_delta"],
        "mean_fold_delta": metrics["mean_fold_delta"] >= gate["minimum_mean_fold_delta"],
        "all_five_folds_win": metrics["winning_folds"] >= gate["minimum_winning_folds"],
        "bad_delta": metrics["categories"][BAD]["delta"] >= gate["minimum_bad_delta"],
        "corrected_to_regressed": metrics["corrected_to_regressed"]
        >= gate["minimum_corrected_to_regressed_ratio"],
        "component_bootstrap": metrics["component_bootstrap"]["probability_delta_positive"]
        >= gate["minimum_component_bootstrap_probability_positive"],
        "flammable_metric_unchanged": abs(metrics["categories"][FLAMMABLE]["delta"])
        <= gate["maximum_flammable_delta_abs"],
        "flammable_fn_nonincrease": metrics["false_negatives"]["flammable"]["delta"]
        <= gate["maximum_flammable_false_negative_increase"],
        "all_positive_fn_nonincrease": metrics["false_negatives"]["all_positive"]["delta"]
        <= gate["maximum_all_positive_false_negative_increase"],
        "flammable_byte_identity": all(
            value for key, value in identity_checks.items() if key.endswith("array_equal")
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "exp682_bad_only_replay_v1",
        "experiment_id": "682",
        "stage": "five_fold_materialized_replay",
        "candidate": "seed632_on_БАД__original_on_Легковоспламеняющиеся",
        "rows": len(labels),
        "folds_evaluated": [0, 1, 2, 3, 4],
        "sealed_rows_loaded": 0,
        "public_used_for_selection": False,
        "input_sha256": {
            "frozen_spec": sha256_file(args.spec),
            "replay_bundle": sha256_file(args.bundle),
            "replay_contract": sha256_file(args.replay_contract),
            "accepted_632_report": sha256_file(args.accepted_632_report),
            "registry": sha256_file(args.registry),
            "exp635_evaluator": sha256_file(args.evaluator),
            "exp635_frozen_spec": sha256_file(args.evaluator.with_name("frozen_spec.json")),
        },
        "frozen_route": frozen["route"],
        "identity_checks": identity_checks,
        "metrics": metrics,
        "gates": gates,
        "validation_signal_accepted": all(gates.values()),
        "gpu_launch_allowed": False,
        "decision": (
            "REPLAY_ACCEPTED_FULL_REFIT_BLOCKED"
            if all(gates.values())
            else "REJECT_REPLAY_SIGNAL"
        ),
    }
    return add_self_hash(report, "report_sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact BAD-only replay on frozen system 140.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replay-contract", type=Path, required=True)
    parser.add_argument("--accepted-632-report", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path(__file__).with_name("frozen_spec.json"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return 0
    if args.output.exists():
        raise FileExistsError("refusing to overwrite replay report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
