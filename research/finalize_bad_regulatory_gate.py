from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
PRIOR_PROTOCOL = (
    "donor-only outer-fold replay of Public-190 exact/name/numeric-family/shingle priors"
)
REQUIRED_BOOLEAN_GATES = (
    "historical_delta_at_least_0_003",
    "historical_wins_at_least_4_of_5",
    "historical_no_category_drop_over_0_005",
    "historical_bootstrap_probability_at_least_0_90",
    "connected_delta_at_least_0_003",
    "connected_wins_at_least_4_of_5",
    "connected_no_category_drop_over_0_005",
    "connected_bootstrap_probability_at_least_0_90",
    "all_repeat_deltas_positive",
    "repeat_mean_delta_at_least_0_003",
    "repeat_wins_at_least_11_of_15",
    "bad_delta_at_least_0_006",
    "sports_delta_at_least_0_01",
    "non_sports_context_drop_at_most_0_005",
    "no_context_cohort_net_regression",
    "corrected_to_regressed_at_least_2",
    "structural_invariants",
    "order_stability_positive_full_and_safe",
    "order_stability_no_category_drop_over_0_005",
    "order_stability_reference_predictions_identical",
)
REQUIRED_NUMERIC_AUDIT_FIELDS = ("repeat_mean_delta", "repeat_fold_wins")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_close(actual: object, expected: object, field: str) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(actual_value) or not math.isfinite(expected_value):
        raise ValueError(f"{field} must be finite")
    if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{field} does not match residual predictions: {actual_value} != {expected_value}"
        )


def finalize_bad_regulatory_gate() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-audit", type=Path, required=True)
    parser.add_argument("--prior-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    residual = json.loads(args.residual_audit.read_text(encoding="utf-8"))
    priors = json.loads(args.prior_replay.read_text(encoding="utf-8"))
    preliminary_key = (
        "accepted_before_downstream_priors_runtime_schema_and_final_refit"
    )
    if preliminary_key not in residual or "acceptance" not in residual:
        raise ValueError("residual audit is missing the fail-closed preliminary verdict")
    if residual.get("experiment_id") != "320":
        raise ValueError("residual audit experiment_id must be 320")
    if residual.get("evaluation_version") != "component_transfer_gate_v4":
        raise ValueError("residual audit evaluation_version must be component_transfer_gate_v4")
    if type(residual[preliminary_key]) is not bool:
        raise ValueError("residual preliminary verdict must be a JSON boolean")
    acceptance = residual["acceptance"]
    if not isinstance(acceptance, dict):
        raise TypeError("residual acceptance must be an object")
    if set(acceptance) != set(REQUIRED_BOOLEAN_GATES) | set(REQUIRED_NUMERIC_AUDIT_FIELDS):
        raise ValueError("residual audit does not contain the exact frozen v4 gate set")
    boolean_gates = {key: acceptance[key] for key in REQUIRED_BOOLEAN_GATES}
    if any(type(value) is not bool for value in boolean_gates.values()):
        raise ValueError("every frozen residual gate must be a JSON boolean")
    if isinstance(acceptance["repeat_mean_delta"], bool) or not isinstance(
        acceptance["repeat_mean_delta"], (int, float)
    ):
        raise TypeError("repeat_mean_delta must be numeric")
    if type(acceptance["repeat_fold_wins"]) is not int:
        raise ValueError("repeat_fold_wins must be an integer")
    if residual[preliminary_key] is not all(boolean_gates.values()):
        raise ValueError("residual preliminary verdict is inconsistent with its gates")
    if set(residual.get("topologies", {})) != {
        "historical",
        "repeat_0",
        "repeat_1",
        "repeat_2",
    }:
        raise ValueError("residual audit must contain the four frozen topologies")
    historical = residual["topologies"]["historical"]
    if priors.get("candidate") != "exp320":
        raise ValueError("prior replay candidate must be exp320")
    if priors.get("protocol") != PRIOR_PROTOCOL:
        raise ValueError("prior replay protocol does not match the frozen downstream stack")
    expected_scores = {
        "baseline_before_priors": {
            "bad_f1": historical["baseline_category_f1"][BAD],
            "flammable_f1": historical["baseline_category_f1"][FLAMMABLE],
            "macro_f1": historical["baseline_macro_f1"],
        },
        "candidate_before_priors": {
            "bad_f1": historical["candidate_category_f1"][BAD],
            "flammable_f1": historical["candidate_category_f1"][FLAMMABLE],
            "macro_f1": historical["candidate_macro_f1"],
        },
    }
    for section, scores in expected_scores.items():
        if not isinstance(priors.get(section), dict) or set(priors[section]) != set(scores):
            raise ValueError(f"prior replay {section} has an invalid score schema")
        for score_name, expected in scores.items():
            require_close(priors[section][score_name], expected, f"{section}.{score_name}")
    if not isinstance(priors.get("delta_before_priors"), dict):
        raise TypeError("prior replay delta_before_priors is missing")
    require_close(
        priors["delta_before_priors"].get("macro_f1"),
        historical["delta_macro_f1"],
        "delta_before_priors.macro_f1",
    )
    survival = float(priors["change_survival"]["survival_rate"])
    post_prior_delta = float(priors["delta_after_priors"]["macro_f1"])
    if not math.isfinite(survival) or not math.isfinite(post_prior_delta):
        raise ValueError("prior replay metrics must be finite")
    prior_gates = {
        "change_survival_at_least_0_8": survival >= 0.8,
        "positive_macro_delta_after_priors": post_prior_delta > 0.0,
    }
    accepted_for_full_refit = residual[preliminary_key] and all(prior_gates.values())
    result = {
        "experiment_id": "320",
        "evaluation_version": "component_transfer_gate_v4",
        "residual_gates": boolean_gates,
        "prior_gates": prior_gates,
        "change_survival_rate": survival,
        "macro_delta_after_priors": post_prior_delta,
        "accepted_for_full_refit": accepted_for_full_refit,
        "production_ready": False,
        "production_blockers": (
            [
                "full-safe-data residual refit and serialized artifact",
                "hidden-schema feature parity smoke",
                "organizer runtime smoke with at least 20 percent reserve",
                "submission package integrity audit",
            ]
            if accepted_for_full_refit
            else ["offline residual or downstream-prior gate failed"]
        ),
        "input_sha256": {
            "residual_audit": sha256(args.residual_audit),
            "prior_replay": sha256(args.prior_replay),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    finalize_bad_regulatory_gate()
