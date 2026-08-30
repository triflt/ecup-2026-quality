from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

INCUMBENT = {
    "nested_macro_f1": 0.9118425205786493,
    "БАД": 0.9517638588912887,
    "Легковоспламеняющиеся": 0.8719211822660099,
}
CANDIDATES = ("hardneg_candidate", "rank_candidate")
CALIBRATION = "category_and_outer_fold_percentile_average_ties"
IMAGE_PREPROCESSING = "solution140_first_image_thumbnail_448_lanczos_v1"
DEPLOYMENT_SCORE = "category_batch_percentile_rank"
THRESHOLD_RULE = "median_of_five_outer_train_thresholds"
MATCHED_CONTROL_VERIFICATION = {
    "schema_version": "exp698_matched_control_verification_v1",
    "fold_output_schema": "exp698_fold_output_v2",
    "common_training_factors_equal_per_fold": True,
    "candidate_teacher_bindings_equal_per_fold": True,
    "gold_teacher_binding_absent": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_self_hash(contract: dict, name: str) -> None:
    payload = dict(contract)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"{name} self-hash mismatch")


def build_selection(
    evaluation: dict,
    guard: dict,
    full_refit: dict | None,
    *,
    evaluation_sha256: str,
    guard_sha256: str,
    full_refit_sha256: str | None,
) -> dict:
    if evaluation.get("schema_version") != "exp698_evaluation_v2":
        raise ValueError("evaluation schema mismatch")
    if evaluation.get("matched_control_verification") != MATCHED_CONTROL_VERIFICATION:
        raise ValueError("matched-control verification mismatch")
    if guard.get("schema_version") != "exp698_connected_family_guard_v2":
        raise ValueError("guard schema mismatch")
    if guard.get("input_sha256", {}).get("evaluation") != evaluation_sha256:
        raise ValueError("guard/evaluation binding mismatch")
    if evaluation.get("image_preprocessing") != IMAGE_PREPROCESSING:
        raise ValueError("evaluation image preprocessing mismatch")
    if evaluation.get("deployment_score") != DEPLOYMENT_SCORE:
        raise ValueError("evaluation deployment score mismatch")
    if evaluation.get("deployment_threshold_rule") != THRESHOLD_RULE:
        raise ValueError("evaluation threshold rule mismatch")

    promoted = guard.get("promoted_candidates")
    if not isinstance(promoted, list) or any(mode not in CANDIDATES for mode in promoted):
        raise ValueError("invalid promoted candidate list")
    if not promoted:
        return {
            "schema_version": "exp718_standalone_selection_v1",
            "experiment_id": "718",
            "authorized": False,
            "decision": "NO_DISTILLATION_CANDIDATE_PROMOTED",
            "evaluation_sha256": evaluation_sha256,
            "connected_guard_sha256": guard_sha256,
            "incumbent_140": INCUMBENT,
            "teacher_required_at_inference": False,
        }
    if full_refit is None or full_refit_sha256 is None:
        raise ValueError("promoted candidate requires full-refit contract")
    verify_self_hash(full_refit, "full-refit contract")
    expected_full = {
        "schema_version": "exp715_full_refit_v1",
        "experiment_id": "715",
        "source_experiment_id": "698",
        "full_data": True,
        "technical_smoke": False,
        "teacher_required_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "image_preprocessing": IMAGE_PREPROCESSING,
    }
    mismatch = {
        key: {"expected": value, "actual": full_refit.get(key)}
        for key, value in expected_full.items()
        if full_refit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"full-refit contract mismatch: {mismatch}")
    binding = full_refit.get("winner_binding", {})
    if (
        binding.get("evaluation_sha256") != evaluation_sha256
        or binding.get("connected_guard_sha256") != guard_sha256
        or binding.get("score_calibration") != CALIBRATION
        or binding.get("image_preprocessing") != IMAGE_PREPROCESSING
    ):
        raise ValueError("full-refit winner binding mismatch")

    selected = max(
        sorted(promoted),
        key=lambda mode: float(evaluation["modes"][mode]["nested_macro_f1"]),
    )
    if full_refit.get("selected_mode") != selected:
        raise ValueError("full-refit selected a different promoted candidate")
    metrics = evaluation.get("modes", {}).get(selected, {})
    if metrics.get("score_calibration") != CALIBRATION:
        raise ValueError("selected candidate score calibration mismatch")
    comparison = evaluation.get("comparisons", {}).get(selected, {})
    audit = guard.get("audits", {}).get(selected, {})
    categories = metrics.get("categories", {})
    if set(categories) != {"БАД", "Легковоспламеняющиеся"}:
        raise ValueError("selected candidate category coverage mismatch")
    thresholds = {}
    category_f1 = {}
    for category in sorted(categories):
        category_f1[category] = float(categories[category]["f1"])
        deployment = categories[category].get("deployment", {})
        threshold = float(deployment.get("threshold", math.nan))
        if (
            deployment.get("score") != DEPLOYMENT_SCORE
            or deployment.get("threshold_rule") != THRESHOLD_RULE
            or not math.isfinite(threshold)
            or not 0.0 < threshold <= 1.0
            or len(deployment.get("outer_train_thresholds", [])) != 5
        ):
            raise ValueError(f"invalid deployment threshold for {category}")
        thresholds[category] = threshold

    macro = float(metrics["nested_macro_f1"])
    gates = {
        "distillation_primary_gate": bool(comparison.get("science_gate")),
        "connected_family_promotion_gate": bool(audit.get("promotion_gate")),
        "macro_delta_vs_incumbent_at_least_0_001": macro - INCUMBENT["nested_macro_f1"] >= 0.001,
        "fold_wins_vs_gold_at_least_4_of_5": int(comparison.get("fold_wins", -1)) >= 4,
        "no_category_drop_vs_incumbent_over_0_005": all(
            category_f1[category] - INCUMBENT[category] >= -0.005
            for category in category_f1
        ),
    }
    authorized = all(gates.values())
    return {
        "schema_version": "exp718_standalone_selection_v1",
        "experiment_id": "718",
        "authorized": authorized,
        "decision": "STANDALONE_4B_AUTHORIZED" if authorized else "STANDALONE_4B_REJECTED",
        "selected_mode": selected,
        "nested_macro_f1": macro,
        "delta_vs_incumbent_140": macro - INCUMBENT["nested_macro_f1"],
        "category_f1": category_f1,
        "deployment_thresholds": thresholds,
        "deployment_score": DEPLOYMENT_SCORE,
        "deployment_threshold_rule": THRESHOLD_RULE,
        "score_calibration": CALIBRATION,
        "image_preprocessing": IMAGE_PREPROCESSING,
        "gates": gates,
        "incumbent_140": INCUMBENT,
        "evaluation_sha256": evaluation_sha256,
        "connected_guard_sha256": guard_sha256,
        "full_refit_contract_sha256": full_refit_sha256,
        "adapter_model_sha256": full_refit["adapter_model_sha256"],
        "teacher_required_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--connected-guard", type=Path, required=True)
    parser.add_argument("--full-refit-contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite standalone selection")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    guard = json.loads(args.connected_guard.read_text(encoding="utf-8"))
    full_refit = (
        json.loads(args.full_refit_contract.read_text(encoding="utf-8"))
        if args.full_refit_contract is not None
        else None
    )
    result = build_selection(
        evaluation,
        guard,
        full_refit,
        evaluation_sha256=sha256(args.evaluation),
        guard_sha256=sha256(args.connected_guard),
        full_refit_sha256=(sha256(args.full_refit_contract) if args.full_refit_contract else None),
    )
    result["contract_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
