from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from build_pair_runtime import canonical_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_PATH = ROOT / "experiments/680_qwen35_4b_flammable_only_hard_bce/evaluate.py"
EXPERIMENT_ID = "685"
FLAMMABLE = "Легковоспламеняющиеся"
STAGE_FOLDS = {
    "outer0": (0,),
    "screen": (0, 3),
    "full": (0, 1, 2, 3, 4),
}


def load_base():
    spec = importlib.util.spec_from_file_location("exp680_for_685", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen experiment-680 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_acceptances(
    scores: list[Path],
    acceptances: list[Path],
    *,
    folds_scope: tuple[int, ...],
    mode: str,
) -> tuple[list[Path], dict[int, dict[str, Any]]]:
    if len(scores) != len(folds_scope) or len(acceptances) != len(folds_scope):
        raise ValueError("score/acceptance count differs from evaluation stage")
    by_fold: dict[int, tuple[Path, dict[str, Any]]] = {}
    for score, acceptance_path in zip(scores, acceptances, strict=True):
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        body = dict(acceptance)
        digest = body.pop("acceptance_sha256", None)
        if digest != canonical_sha256(body):
            raise ValueError("training acceptance self-hash mismatch")
        fold = int(acceptance.get("outer_fold", -1))
        expected = {
            "experiment_id": EXPERIMENT_ID,
            "mode": mode,
            "technical_smoke": False,
            "deployable_4b_only": True,
            "uses_27b_at_inference": False,
            "validation_labels_read": 0,
            "sealed_rows_used": 0,
            "public_used": False,
            "decision": "ACCEPT",
            "predictions_sha256": sha256_file(score),
            "rank_loss_weight": 0.5 if mode == "rank_candidate" else 0.0,
        }
        if any(acceptance.get(key) != value for key, value in expected.items()):
            raise ValueError("prediction provenance or training mode mismatch")
        if fold not in folds_scope or fold in by_fold:
            raise ValueError("prediction fold mismatch or duplicate")
        by_fold[fold] = (score, acceptance)
    if set(by_fold) != set(folds_scope):
        raise ValueError("prediction folds do not exactly cover stage")
    return (
        [by_fold[fold][0] for fold in folds_scope],
        {fold: by_fold[fold][1] for fold in folds_scope},
    )


def verify_paired_contracts(
    control: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> None:
    parity_fields = (
        "rows",
        "pairs",
        "pair_runtime_contract_sha256",
        "pair_runtime_acceptance_sha256",
        "transport_acceptance_sha256",
        "source_641_runtime_contract_sha256",
        "code_bundle_sha256",
        "code_revision",
        "code_manifest_sha256",
        "code_acceptance_sha256",
        "vendor_zip_sha256",
        "vendor_bridge_sha256",
        "vendor_source_bundle_sha256",
        "exact_runtime_binding",
        "model_id",
        "model_revision",
        "model_tree_sha256",
        "model_tree_files",
        "initial_trainable_state_sha256",
        "ordered_pair_index_sha256",
        "seed",
        "epochs",
        "learning_rate",
        "micro_batch_pairs",
        "micro_batch_rows",
        "gradient_accumulation_pairs",
        "effective_batch_rows",
        "optimizer_steps_executed",
        "runtime_backend",
        "runtime_packages",
    )
    for fold, control_acceptance in control.items():
        if any(
            control_acceptance.get(key) != candidate[fold].get(key)
            for key in parity_fields
        ):
            raise ValueError("control/candidate runtime parity mismatch")


def correction_ratio_ok(metrics: dict[str, Any], minimum: float) -> bool:
    return (
        metrics["corrected"] > 0
        if metrics["regressed"] == 0
        else metrics["corrected_to_regressed"] >= minimum
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite evaluation")
    folds_scope = STAGE_FOLDS[args.stage]
    control_scores, control_acceptances = load_acceptances(
        args.control_score,
        args.control_acceptance,
        folds_scope=folds_scope,
        mode="paired_hard_control",
    )
    candidate_scores, candidate_acceptances = load_acceptances(
        args.candidate_score,
        args.candidate_acceptance,
        folds_scope=folds_scope,
        mode="rank_candidate",
    )
    verify_paired_contracts(control_acceptances, candidate_acceptances)
    if (
        sha256_file(args.bundle) != BASE.FROZEN_BUNDLE_SHA256
        or sha256_file(args.replay_contract) != BASE.FROZEN_CONTRACT_SHA256
        or sha256_file(args.registry) != BASE.FROZEN_REGISTRY_SHA256
    ):
        raise ValueError("frozen replay input checksum mismatch")
    spec = BASE.LEGACY.load_spec()
    BASE.LEGACY.verify_source_recipe(spec)
    contract = BASE.LEGACY.verify_replay_contract(
        path=args.replay_contract,
        bundle_path=args.bundle,
        registry_path=args.registry,
        spec=spec,
    )
    registry = BASE.LEGACY._load_registry(args.registry, spec=spec)
    bundle = BASE.LEGACY.load_replay_bundle(
        path=args.bundle, contract=contract, registry=registry
    )
    candidate_logits, candidate_hashes = BASE.load_scores(
        candidate_scores, bundle=bundle, folds_scope=folds_scope
    )
    control_logits, control_hashes = BASE.load_scores(
        control_scores, bundle=bundle, folds_scope=folds_scope
    )
    labels = registry["label"].astype(np.int8).to_numpy()
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    components = bundle["semantic_components"].astype(str)
    selected = np.isin(folds, folds_scope) & (categories == FLAMMABLE)
    fold_ap: dict[str, Any] = {}
    for fold in folds_scope:
        local = selected & (folds == fold)
        before = BASE.average_precision(labels[local], control_logits[local])
        after = BASE.average_precision(labels[local], candidate_logits[local])
        fold_ap[str(fold)] = {
            "control": before,
            "candidate": after,
            "delta": after - before,
        }
    baseline_probability = BASE.LEGACY.sigmoid(bundle["qwen35_original_logit"])
    control_probability = baseline_probability.copy()
    candidate_probability = baseline_probability.copy()
    flammable = categories == FLAMMABLE
    control_probability[flammable] = BASE.LEGACY.sigmoid(control_logits[flammable])
    candidate_probability[flammable] = BASE.LEGACY.sigmoid(candidate_logits[flammable])
    baseline_prediction, _ = BASE.LEGACY.route_predictions(
        bundle=bundle, qwen_probability=baseline_probability, route=spec["route"]
    )
    control_prediction, _ = BASE.LEGACY.route_predictions(
        bundle=bundle, qwen_probability=control_probability, route=spec["route"]
    )
    candidate_prediction, _ = BASE.LEGACY.route_predictions(
        bundle=bundle, qwen_probability=candidate_probability, route=spec["route"]
    )
    bootstrap = spec["bootstrap"] if args.stage == "full" else None
    direct_metrics = BASE.LEGACY._metrics(
        labels=labels,
        categories=categories,
        folds=folds,
        components=components,
        baseline=control_prediction,
        candidate=candidate_prediction,
        selected_folds=folds_scope,
        bootstrap=bootstrap,
    )
    production_metrics = BASE.LEGACY._metrics(
        labels=labels,
        categories=categories,
        folds=folds,
        components=components,
        baseline=baseline_prediction,
        candidate=candidate_prediction,
        selected_folds=folds_scope,
        bootstrap=bootstrap,
    )
    component_sizes = Counter(components.tolist())
    singleton = selected & np.array(
        [component_sizes[value] == 1 for value in components]
    )
    singleton_corrected = int(
        np.sum(singleton & (control_prediction != labels) & (candidate_prediction == labels))
    )
    singleton_regressed = int(
        np.sum(singleton & (control_prediction == labels) & (candidate_prediction != labels))
    )
    ap_deltas = [fold_ap[str(fold)]["delta"] for fold in folds_scope]
    common = {
        "bad_route_byte_identical": (
            direct_metrics["categories"]["БАД"]["delta"] == 0.0
            and production_metrics["categories"]["БАД"]["delta"] == 0.0
        ),
        "flammable_false_negatives_do_not_increase": (
            direct_metrics["false_negatives"]["flammable"]["delta"] <= 0
        ),
    }
    if args.stage == "outer0":
        gates = common | {
            "outer0_ap_positive": ap_deltas[0] > 0.0,
            "outer0_direct_macro_nonnegative": direct_metrics["mean_fold_delta"] >= 0.0,
            "outer0_corrections_to_regressions_at_least_1_2": correction_ratio_ok(
                direct_metrics, 1.2
            ),
        }
        decision = "OPEN_SCREEN_FOLD3" if all(gates.values()) else "REJECT_AT_OUTER0"
    elif args.stage == "screen":
        gates = common | {
            "both_folds_ap_positive": all(value > 0.0 for value in ap_deltas),
            "mean_ap_delta_at_least_0_010": float(np.mean(ap_deltas)) >= 0.010,
            "both_folds_direct_macro_positive": direct_metrics["winning_folds"] == 2,
            "mean_direct_macro_delta_at_least_0_003": (
                direct_metrics["mean_fold_delta"] >= 0.003
            ),
            "direct_flammable_f1_delta_at_least_0_010": (
                direct_metrics["categories"][FLAMMABLE]["delta"] >= 0.010
            ),
            "corrections_to_regressions_at_least_1_5": correction_ratio_ok(
                direct_metrics, 1.5
            ),
            "semantic_singleton_net_positive": (
                singleton_corrected - singleton_regressed > 0
            ),
            "production_macro_nonnegative": production_metrics["mean_fold_delta"] >= 0.0,
        }
        decision = "OPEN_CONFIRMATION" if all(gates.values()) else "REJECT_AT_SCREEN"
    else:
        gates = common | {
            "at_least_four_of_five_ap_wins": sum(value > 0.0 for value in ap_deltas) >= 4,
            "mean_ap_delta_at_least_0_010": float(np.mean(ap_deltas)) >= 0.010,
            "confirmation_direct_macro_positive": all(
                direct_metrics["folds"][str(fold)]["delta"] > 0.0
                for fold in (1, 2, 4)
            ),
            "at_least_four_of_five_direct_macro_wins": (
                direct_metrics["winning_folds"] >= 4
            ),
            "mean_direct_macro_delta_at_least_0_006": (
                direct_metrics["mean_fold_delta"] >= 0.006
            ),
            "direct_flammable_f1_delta_at_least_0_012": (
                direct_metrics["categories"][FLAMMABLE]["delta"] >= 0.012
            ),
            "corrections_to_regressions_at_least_1_5": correction_ratio_ok(
                direct_metrics, 1.5
            ),
            "bootstrap_probability_at_least_0_90": (
                direct_metrics["component_bootstrap"]["probability_delta_positive"]
                >= 0.90
            ),
            "semantic_singleton_net_positive": (
                singleton_corrected - singleton_regressed > 0
            ),
            "production_macro_delta_at_least_0_006": (
                production_metrics["macro_delta"] >= 0.006
            ),
            "production_flammable_f1_delta_at_least_0_012": (
                production_metrics["categories"][FLAMMABLE]["delta"] >= 0.012
            ),
        }
        decision = "ACCEPT_FOR_REFIT" if all(gates.values()) else "REJECT_FULL"
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "folds": list(folds_scope),
        "changed_factor": "add_fixed_rank_loss_weight_0.5_to_paired_hard_control",
        "fold_flammable_average_precision": fold_ap,
        "mean_flammable_average_precision_delta": float(np.mean(ap_deltas)),
        "direct_control_metrics": direct_metrics,
        "production_baseline_metrics": production_metrics,
        "semantic_singleton_flammable": {
            "rows": int(singleton.sum()),
            "corrected": singleton_corrected,
            "regressed": singleton_regressed,
            "net": singleton_corrected - singleton_regressed,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "decision": decision,
        "candidate_prediction_sha256": candidate_hashes,
        "control_prediction_sha256": control_hashes,
        "candidate_acceptance_sha256": [
            candidate_acceptances[fold]["acceptance_sha256"] for fold in folds_scope
        ],
        "control_acceptance_sha256": [
            control_acceptances[fold]["acceptance_sha256"] for fold in folds_scope
        ],
        "validation_labels_read_by_training": 0,
        "validation_labels_read_by_evaluator": int(selected.sum()),
        "sealed_rows": 0,
        "public_used": False,
        "threshold_tuned": False,
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
    parser.add_argument("--stage", choices=tuple(STAGE_FOLDS), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replay-contract", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-score", type=Path, action="append", required=True)
    parser.add_argument(
        "--candidate-acceptance", type=Path, action="append", required=True
    )
    parser.add_argument("--control-score", type=Path, action="append", required=True)
    parser.add_argument(
        "--control-acceptance", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(evaluate(arguments), ensure_ascii=False, indent=2, sort_keys=True))
