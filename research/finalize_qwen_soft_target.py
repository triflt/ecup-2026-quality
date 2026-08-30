from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked", type=Path, required=True)
    parser.add_argument("--connected", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--sports", type=Path, required=True)
    parser.add_argument("--candidate", default="exp310")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    locked_file = json.loads(args.locked.read_text(encoding="utf-8"))
    connected_file = json.loads(args.connected.read_text(encoding="utf-8"))
    priors = json.loads(args.priors.read_text(encoding="utf-8"))
    sports = json.loads(args.sports.read_text(encoding="utf-8"))
    locked = locked_file["candidates"][args.candidate]
    connected = connected_file["audit"]["connected_safe_rows"]
    locked_category_delta = locked["category_delta"]
    connected_category_delta = {
        category: connected["candidate"][category] - connected["baseline"][category]
        for category in (BAD, FLAMMABLE)
    }
    gates = {
        "locked_delta_at_least_0_003": locked["delta_macro_f1"] >= 0.003,
        "locked_wins_at_least_4_of_5": locked["folds_won"] >= 4,
        "locked_no_category_drop_over_0_005": min(locked_category_delta.values())
        >= -0.005,
        "locked_bootstrap_probability_at_least_0_90": locked["group_bootstrap"][
            "probability_delta_positive"
        ]
        >= 0.90,
        "connected_delta_at_least_0_003": connected["delta_macro_f1"] >= 0.003,
        "connected_wins_at_least_4_of_5": connected["folds_won"] >= 4,
        "connected_no_category_drop_over_0_005": min(
            connected_category_delta.values()
        )
        >= -0.005,
        "connected_bootstrap_probability_at_least_0_90": connected[
            "component_bootstrap"
        ]["probability_delta_positive"]
        >= 0.90,
        "downstream_change_survival_at_least_0_8": priors["change_survival"][
            "survival_rate"
        ]
        >= 0.8,
        "positive_macro_delta_after_priors": priors["delta_after_priors"][
            "macro_f1"
        ]
        > 0.0,
    }
    offline_core_passed = all(gates.values())
    fixed = int(sports["fixed_from_frozen_243"])
    guardrail_breaches = {
        "sports_new_errors_exceed_corrections": bool(sports["guardrail_breach"]),
        "locked_category_drop_over_0_005": min(locked_category_delta.values())
        < -0.005,
        "connected_category_drop_over_0_005": min(
            connected_category_delta.values()
        )
        < -0.005,
        "nonpositive_macro_delta_after_priors": priors["delta_after_priors"][
            "macro_f1"
        ]
        <= 0.0,
    }
    guardrail_breach = any(guardrail_breaches.values())
    launch_320 = not (fixed >= 49 and not guardrail_breach)
    evidence_paths = [args.locked, args.connected, args.sports, args.priors]
    result = {
        "audit_complete": True,
        "offline_core_gate_passed": offline_core_passed,
        "sports_errors_fixed_out_of_243": fixed,
        "guardrail_breach": guardrail_breach,
        "guardrail_breaches": guardrail_breaches,
        "launch_experiment_320": launch_320,
        "gates": gates,
        "locked_summary": {
            "baseline_macro_f1": locked["baseline_macro_f1"],
            "candidate_macro_f1": locked["candidate_macro_f1"],
            "delta_macro_f1": locked["delta_macro_f1"],
            "folds_won": locked["folds_won"],
            "category_delta": locked_category_delta,
            "bootstrap_probability_delta_positive": locked["group_bootstrap"][
                "probability_delta_positive"
            ],
        },
        "connected_summary": {
            "delta_macro_f1": connected["delta_macro_f1"],
            "folds_won": connected["folds_won"],
            "category_delta": connected_category_delta,
            "bootstrap_probability_delta_positive": connected[
                "component_bootstrap"
            ]["probability_delta_positive"],
        },
        "priors_summary": {
            "change_survival_rate": priors["change_survival"]["survival_rate"],
            "macro_delta_after_priors": priors["delta_after_priors"]["macro_f1"],
        },
        "sports_summary": sports,
        "evidence": [
            {"path": str(path), "sha256": sha256(path)} for path in evidence_paths
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
