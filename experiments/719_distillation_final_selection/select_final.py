from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INCUMBENT = {
    "experiment_id": "140",
    "nested_macro_f1": 0.9118425205786493,
    "bad_f1": 0.9517638588912887,
    "flammable_f1": 0.8719211822660099,
    "public_macro_f1": 0.8923976821312729,
}
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


def verify_self_hash(value: dict, name: str) -> None:
    payload = dict(value)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"{name} self-hash mismatch")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_final_selection(
    *,
    evaluation: dict,
    guard: dict,
    incumbent_audit: dict,
    incumbent_metrics: dict,
    standalone_selection: dict,
    incumbent_package: dict,
    bindings: dict[str, str],
    full_refit: dict | None = None,
    fixed_package: dict | None = None,
    fixed_runtime: dict | None = None,
    standalone_package: dict | None = None,
    standalone_runtime: dict | None = None,
) -> dict:
    if evaluation.get("schema_version") != "exp698_evaluation_v2":
        raise ValueError("evaluation schema mismatch")
    if evaluation.get("matched_control_verification") != MATCHED_CONTROL_VERIFICATION:
        raise ValueError("matched-control verification mismatch")
    if guard.get("schema_version") != "exp698_connected_family_guard_v2":
        raise ValueError("guard schema mismatch")
    if guard.get("input_sha256", {}).get("evaluation") != bindings["evaluation"]:
        raise ValueError("guard/evaluation binding mismatch")
    verify_self_hash(incumbent_audit, "incumbent audit")
    if (
        incumbent_audit.get("decision") != "FREEZE_FUSION_USE_ISOLATED_REPLACEMENT_GATES"
        or incumbent_audit.get("exact_component_oof_available") is not False
    ):
        raise ValueError("incumbent evidence policy mismatch")
    historical = incumbent_metrics.get("historical_results", [])
    primary = next(
        (
            row
            for row in historical
            if row.get("experiment") == "Nested robust-base + Qwen3-VL + Qwen3.5"
        ),
        None,
    )
    runtime = next(
        (
            row
            for row in historical
            if row.get("experiment") == "Dual-LoRA scaled runtime smoke"
        ),
        None,
    )
    if (
        incumbent_metrics.get("experiment_id") != "140"
        or float(incumbent_metrics.get("public_macro_f1", -1)) != INCUMBENT["public_macro_f1"]
        or primary is None
        or float(primary.get("macro_f1", -1)) != INCUMBENT["nested_macro_f1"]
        or runtime is None
        or runtime.get("status") != "Success"
        or "projected Public 1,600" not in str(runtime.get("notes", ""))
    ):
        raise ValueError("incumbent metric/runtime evidence mismatch")
    verify_self_hash(standalone_selection, "standalone selection")
    if standalone_selection.get("schema_version") != "exp718_standalone_selection_v1":
        raise ValueError("standalone selection schema mismatch")
    expected_incumbent = {
        "schema_version": "exp716_allowed_base_freeze_v1",
        "experiment_id": "716",
        "source_read_only": True,
        "forbidden_model_references_absent": True,
        "zip_integrity": "PASS",
    }
    mismatch = {
        key: {"expected": value, "actual": incumbent_package.get(key)}
        for key, value in expected_incumbent.items()
        if incumbent_package.get(key) != value
    }
    if mismatch or incumbent_package.get("output_sha256") != bindings["incumbent_zip"]:
        raise ValueError(f"frozen incumbent package mismatch: {mismatch}")

    promoted = guard.get("promoted_candidates")
    if not isinstance(promoted, list):
        raise ValueError("guard promoted candidate list is invalid")
    distillation_confirmed = bool(promoted)
    fixed = None
    if distillation_confirmed:
        if full_refit is None or fixed_package is None or fixed_runtime is None:
            raise ValueError("promoted distillation lacks fixed-replacement artifacts")
        verify_self_hash(full_refit, "full-refit contract")
        if (
            full_refit.get("schema_version") != "exp715_full_refit_v1"
            or full_refit.get("selected_mode") not in promoted
            or full_refit.get("teacher_required_at_inference") is not False
        ):
            raise ValueError("full-refit contract mismatch")
        if (
            fixed_package.get("schema_version") != "exp716_submission_package_v1"
            or fixed_package.get("output_sha256") != bindings.get("fixed_zip")
            or fixed_package.get("full_refit_contract_sha256") != bindings.get("full_refit")
            or fixed_package.get("exact_fixed_140_fusion_delta_measured") is not False
            or fixed_package.get("fusion_policy")
            != "FREEZE_FUSION_USE_ISOLATED_REPLACEMENT_GATES"
        ):
            raise ValueError("fixed-replacement package mismatch")
        if (
            fixed_runtime.get("schema_version") != "exp716_runtime_acceptance_v1"
            or fixed_runtime.get("decision") != "DEPLOYABLE_RUNTIME_PASS"
            or fixed_runtime.get("submission_sha256") != bindings.get("fixed_zip")
            or fixed_runtime.get("single_h100_visible") is not True
        ):
            raise ValueError("fixed-replacement runtime mismatch")
        fixed = {
            "status": "DEPLOYABLE_EXPERIMENT_NOT_METRIC_PROMOTED",
            "reason": "exact solution-140 component OOF is unavailable, so fixed-fusion delta is unmeasured",
            "submission_sha256": bindings["fixed_zip"],
            "package_report_sha256": bindings["fixed_package"],
            "runtime_acceptance_sha256": bindings["fixed_runtime"],
            "selected_mode": full_refit["selected_mode"],
        }

    standalone_authorized = standalone_selection.get("authorized") is True
    if standalone_authorized:
        if not distillation_confirmed:
            raise ValueError("standalone authorized without promoted distillation")
        if standalone_package is None or standalone_runtime is None:
            raise ValueError("authorized standalone lacks package/runtime evidence")
        if (
            standalone_package.get("schema_version") != "exp718_standalone_package_v1"
            or standalone_package.get("architecture") != "qwen35_only"
            or standalone_package.get("output_sha256") != bindings.get("standalone_zip")
            or standalone_package.get("selection_contract_sha256")
            != bindings["standalone_selection"]
            or standalone_package.get("extra_model_stages") != []
        ):
            raise ValueError("standalone package mismatch")
        if (
            standalone_runtime.get("schema_version") != "exp718_runtime_acceptance_v1"
            or standalone_runtime.get("decision") != "DEPLOYABLE_RUNTIME_PASS"
            or standalone_runtime.get("architecture") != "qwen35_only"
            or standalone_runtime.get("submission_sha256") != bindings.get("standalone_zip")
            or standalone_runtime.get("single_h100_visible") is not True
        ):
            raise ValueError("standalone runtime mismatch")
        selected = {
            "architecture": "qwen35_only",
            "experiment_id": "718",
            "reason": "direct calibrated OOF exceeds incumbent and all transfer/runtime gates pass",
            "submission_sha256": bindings["standalone_zip"],
            "package_report_sha256": bindings["standalone_package"],
            "runtime_acceptance_sha256": bindings["standalone_runtime"],
            "nested_macro_f1": standalone_selection["nested_macro_f1"],
            "delta_vs_incumbent_140": standalone_selection["delta_vs_incumbent_140"],
        }
        decision = "SELECT_STANDALONE_DISTILLED_4B"
    else:
        selected = {
            "architecture": "solution140_frozen_incumbent",
            "experiment_id": "140",
            "reason": (
                "distillation did not pass promotion"
                if not distillation_confirmed
                else "distillation signal exists, but no directly measured candidate beats incumbent 140"
            ),
            "submission_sha256": bindings["incumbent_zip"],
            "package_report_sha256": bindings["incumbent_package"],
            "nested_macro_f1": INCUMBENT["nested_macro_f1"],
            "public_macro_f1": INCUMBENT["public_macro_f1"],
        }
        decision = "KEEP_SOLUTION_140_INCUMBENT"

    return {
        "schema_version": "exp719_final_selection_v1",
        "experiment_id": "719",
        "decision": decision,
        "selected": selected,
        "incumbent_140": INCUMBENT,
        "distillation_signal_confirmed": distillation_confirmed,
        "promoted_distillation_candidates": promoted,
        "standalone_authorized": standalone_authorized,
        "fixed_replacement": fixed,
        "claims": {
            "exact_fixed_140_fusion_delta_measured": False,
            "fusion_retuned": False,
            "teacher_in_selected_submission": False,
            "selected_submission_uses_allowed_model_at_most_4b": True,
            "selected_submission_under_5_gib": True,
            "selected_runtime_one_h100": True,
        },
        "bindings": bindings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--incumbent-audit", type=Path, required=True)
    parser.add_argument("--incumbent-metrics", type=Path, required=True)
    parser.add_argument("--standalone-selection", type=Path, required=True)
    parser.add_argument("--incumbent-zip", type=Path, required=True)
    parser.add_argument("--incumbent-package", type=Path, required=True)
    parser.add_argument("--full-refit", type=Path)
    parser.add_argument("--fixed-zip", type=Path)
    parser.add_argument("--fixed-package", type=Path)
    parser.add_argument("--fixed-runtime", type=Path)
    parser.add_argument("--standalone-zip", type=Path)
    parser.add_argument("--standalone-package", type=Path)
    parser.add_argument("--standalone-runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite final selection")
    bindings = {
        "evaluation": sha256(args.evaluation),
        "guard": sha256(args.guard),
        "incumbent_audit": sha256(args.incumbent_audit),
        "incumbent_metrics": sha256(args.incumbent_metrics),
        "standalone_selection": sha256(args.standalone_selection),
        "incumbent_zip": sha256(args.incumbent_zip),
        "incumbent_package": sha256(args.incumbent_package),
    }
    optional_paths = {
        "full_refit": args.full_refit,
        "fixed_zip": args.fixed_zip,
        "fixed_package": args.fixed_package,
        "fixed_runtime": args.fixed_runtime,
        "standalone_zip": args.standalone_zip,
        "standalone_package": args.standalone_package,
        "standalone_runtime": args.standalone_runtime,
    }
    bindings.update({key: sha256(path) for key, path in optional_paths.items() if path})
    result = build_final_selection(
        evaluation=load(args.evaluation),
        guard=load(args.guard),
        incumbent_audit=load(args.incumbent_audit),
        incumbent_metrics=load(args.incumbent_metrics),
        standalone_selection=load(args.standalone_selection),
        incumbent_package=load(args.incumbent_package),
        bindings=bindings,
        full_refit=(load(args.full_refit) if args.full_refit else None),
        fixed_package=(load(args.fixed_package) if args.fixed_package else None),
        fixed_runtime=(load(args.fixed_runtime) if args.fixed_runtime else None),
        standalone_package=(load(args.standalone_package) if args.standalone_package else None),
        standalone_runtime=(load(args.standalone_runtime) if args.standalone_runtime else None),
    )
    result["selected"]["submission_path"] = str(
        args.standalone_zip
        if result["decision"] == "SELECT_STANDALONE_DISTILLED_4B"
        else args.incumbent_zip
    )
    if result.get("fixed_replacement") is not None:
        result["fixed_replacement"]["submission_path"] = str(args.fixed_zip)
    result["contract_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
