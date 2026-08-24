from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT_PATH = ROOT / "experiments/680_qwen35_4b_flammable_only_hard_bce/evaluate.py"
FLAMMABLE = "Легковоспламеняющиеся"


def load_parent():
    spec = importlib.util.spec_from_file_location("exp680_for_681", PARENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen experiment-680 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARENT = load_parent()


def singleton_slice(args: argparse.Namespace) -> dict[str, Any]:
    folds_scope = PARENT.SCREEN_FOLDS if args.stage == "screen" else PARENT.FULL_FOLDS
    spec = PARENT.LEGACY.load_spec()
    PARENT.LEGACY.verify_source_recipe(spec)
    contract = PARENT.LEGACY.verify_replay_contract(
        path=args.replay_contract,
        bundle_path=args.bundle,
        registry_path=args.registry,
        spec=spec,
    )
    registry = PARENT.LEGACY._load_registry(args.registry, spec=spec)
    bundle = PARENT.LEGACY.load_replay_bundle(
        path=args.bundle, contract=contract, registry=registry
    )
    logits, _ = PARENT.load_scores(
        args.candidate_score, bundle=bundle, folds_scope=folds_scope
    )
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    components = bundle["semantic_components"].astype(str)
    labels = registry["label"].astype(np.int8).to_numpy()
    unique, counts = np.unique(components, return_counts=True)
    singleton_components = set(unique[counts == 1])
    mask = (
        np.isin(folds, folds_scope)
        & (categories == FLAMMABLE)
        & np.isin(components, list(singleton_components))
    )
    baseline_prob = PARENT.LEGACY.sigmoid(bundle["qwen35_original_logit"])
    candidate_prob = baseline_prob.copy()
    candidate_prob[categories == FLAMMABLE] = PARENT.LEGACY.sigmoid(
        logits[categories == FLAMMABLE]
    )
    baseline, _ = PARENT.LEGACY.route_predictions(
        bundle=bundle, qwen_probability=baseline_prob, route=spec["route"]
    )
    candidate, _ = PARENT.LEGACY.route_predictions(
        bundle=bundle, qwen_probability=candidate_prob, route=spec["route"]
    )
    corrected = int(np.sum(mask & (baseline != labels) & (candidate == labels)))
    regressed = int(np.sum(mask & (baseline == labels) & (candidate != labels)))
    before = PARENT.LEGACY.f1(labels[mask], baseline[mask])
    after = PARENT.LEGACY.f1(labels[mask], candidate[mask])
    return {
        "rows": int(mask.sum()),
        "baseline_flammable_f1": before,
        "candidate_flammable_f1": after,
        "flammable_f1_delta": after - before,
        "corrected": corrected,
        "regressed": regressed,
        "net_corrections": corrected - regressed,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite evaluation")
    with tempfile.TemporaryDirectory(prefix="exp681_eval_") as directory:
        parent_output = Path(directory) / "parent.json"
        parent_values = dict(vars(args))
        parent_values["output"] = parent_output
        parent_args = SimpleNamespace(**parent_values)
        result = PARENT.evaluate(parent_args)
    result["experiment_id"] = "681"
    result["changed_factor"] = "hard_bce_to_fixed_hard_plus_teacher_soft_bce"
    result["temperature"] = 2.0
    result["soft_loss_weight"] = 0.5
    result["ordinary_oof_merge_used"] = False
    result["uses_27b_at_inference"] = False
    singleton = singleton_slice(args)
    result["semantic_singleton_flammable"] = singleton
    if args.stage == "screen":
        # Experiment 681 preregistered no relaxed screen continuation.  The
        # parent evaluator exposes an ablation screen for its own experiment;
        # carrying that decision forward here would silently weaken the gate.
        result["ablation_submission_eligible"] = False
        result["acceptance_tier"] = "primary" if result["passed"] else "no_go"
        result["decision"] = (
            "OPEN_CONFIRMATION" if result["passed"] else "REJECT_AT_SCREEN"
        )
    else:
        primary = result["gates"]
        fallback = result["ablation_gates"]
        primary["mean_macro_delta_at_least_0_006"] = (
            result["production_metrics"]["mean_fold_delta"] >= 0.006
        )
        primary["singleton_net_corrections_positive"] = singleton["net_corrections"] > 0
        fallback["mean_macro_delta_at_least_0_004"] = (
            result["production_metrics"]["mean_fold_delta"] >= 0.004
        )
        fallback["singleton_net_corrections_positive"] = singleton["net_corrections"] > 0
        result["passed"] = all(primary.values())
        result["ablation_submission_eligible"] = all(fallback.values())
        if result["passed"]:
            result["acceptance_tier"] = "primary"
            result["decision"] = "ACCEPT_FOR_REFIT"
        elif result["ablation_submission_eligible"]:
            result["acceptance_tier"] = "public_ablation"
            result["decision"] = "ACCEPT_FOR_REFIT"
        else:
            result["acceptance_tier"] = "no_go"
            result["decision"] = "REJECT_FULL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    arguments = parser.parse_args()
    print(json.dumps(evaluate(arguments), ensure_ascii=False, indent=2, sort_keys=True))
