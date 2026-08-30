from __future__ import annotations

import unittest

from select_final import (
    MATCHED_CONTROL_VERIFICATION,
    build_final_selection,
    canonical_sha256,
)


def base() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    evaluation = {
        "schema_version": "exp698_evaluation_v2",
        "matched_control_verification": MATCHED_CONTROL_VERIFICATION,
    }
    guard = {
        "schema_version": "exp698_connected_family_guard_v2",
        "input_sha256": {"evaluation": "e"},
        "promoted_candidates": [],
    }
    audit = {
        "decision": "FREEZE_FUSION_USE_ISOLATED_REPLACEMENT_GATES",
        "exact_component_oof_available": False,
    }
    audit["contract_sha256"] = canonical_sha256(audit)
    standalone = {
        "schema_version": "exp718_standalone_selection_v1",
        "authorized": False,
        "decision": "NO_DISTILLATION_CANDIDATE_PROMOTED",
    }
    standalone["contract_sha256"] = canonical_sha256(standalone)
    incumbent = {
        "schema_version": "exp716_allowed_base_freeze_v1",
        "experiment_id": "716",
        "source_read_only": True,
        "forbidden_model_references_absent": True,
        "zip_integrity": "PASS",
        "output_sha256": "i",
    }
    metrics = {
        "experiment_id": "140",
        "public_macro_f1": 0.8923976821312729,
        "historical_results": [
            {
                "experiment": "Nested robust-base + Qwen3-VL + Qwen3.5",
                "macro_f1": "0.9118425205786493",
            },
            {
                "experiment": "Dual-LoRA scaled runtime smoke",
                "status": "Success",
                "notes": "projected Public 1,600 = 10.17 min",
            },
        ],
    }
    bindings = {
        "evaluation": "e",
        "guard": "g",
        "incumbent_audit": "a",
        "incumbent_metrics": "m",
        "standalone_selection": "s",
        "incumbent_zip": "i",
        "incumbent_package": "p",
    }
    return evaluation, guard, audit, metrics, standalone, incumbent, bindings


class FinalSelectionTest(unittest.TestCase):
    def test_rejects_stale_evaluation_without_matched_control_evidence(self) -> None:
        evaluation, guard, audit, metrics, standalone, incumbent, bindings = base()
        evaluation.pop("matched_control_verification")
        with self.assertRaisesRegex(ValueError, "matched-control verification mismatch"):
            build_final_selection(
                evaluation=evaluation,
                guard=guard,
                incumbent_audit=audit,
                incumbent_metrics=metrics,
                standalone_selection=standalone,
                incumbent_package=incumbent,
                bindings=bindings,
            )

    def test_keeps_incumbent_when_distillation_not_promoted(self) -> None:
        evaluation, guard, audit, metrics, standalone, incumbent, bindings = base()
        result = build_final_selection(
            evaluation=evaluation,
            guard=guard,
            incumbent_audit=audit,
            incumbent_metrics=metrics,
            standalone_selection=standalone,
            incumbent_package=incumbent,
            bindings=bindings,
        )
        self.assertEqual(result["decision"], "KEEP_SOLUTION_140_INCUMBENT")
        self.assertFalse(result["distillation_signal_confirmed"])

    def test_selects_runtime_verified_standalone(self) -> None:
        evaluation, guard, audit, metrics, standalone, incumbent, bindings = base()
        guard["promoted_candidates"] = ["hardneg_candidate"]
        full = {
            "schema_version": "exp715_full_refit_v1",
            "selected_mode": "hardneg_candidate",
            "teacher_required_at_inference": False,
        }
        full["contract_sha256"] = canonical_sha256(full)
        standalone.update(
            {
                "authorized": True,
                "decision": "STANDALONE_4B_AUTHORIZED",
                "nested_macro_f1": 0.914,
                "delta_vs_incumbent_140": 0.0021574794213507,
            }
        )
        standalone.pop("contract_sha256")
        standalone["contract_sha256"] = canonical_sha256(standalone)
        bindings.update(
            {
                "full_refit": "f",
                "fixed_zip": "z",
                "fixed_package": "fp",
                "fixed_runtime": "fr",
                "standalone_zip": "q",
                "standalone_package": "sp",
                "standalone_runtime": "sr",
            }
        )
        fixed_package = {
            "schema_version": "exp716_submission_package_v1",
            "output_sha256": "z",
            "full_refit_contract_sha256": "f",
            "exact_fixed_140_fusion_delta_measured": False,
            "fusion_policy": "FREEZE_FUSION_USE_ISOLATED_REPLACEMENT_GATES",
        }
        fixed_runtime = {
            "schema_version": "exp716_runtime_acceptance_v1",
            "decision": "DEPLOYABLE_RUNTIME_PASS",
            "submission_sha256": "z",
            "single_h100_visible": True,
        }
        standalone_package = {
            "schema_version": "exp718_standalone_package_v1",
            "architecture": "qwen35_only",
            "output_sha256": "q",
            "selection_contract_sha256": "s",
            "extra_model_stages": [],
        }
        standalone_runtime = {
            "schema_version": "exp718_runtime_acceptance_v1",
            "decision": "DEPLOYABLE_RUNTIME_PASS",
            "architecture": "qwen35_only",
            "submission_sha256": "q",
            "single_h100_visible": True,
        }
        result = build_final_selection(
            evaluation=evaluation,
            guard=guard,
            incumbent_audit=audit,
            incumbent_metrics=metrics,
            standalone_selection=standalone,
            incumbent_package=incumbent,
            bindings=bindings,
            full_refit=full,
            fixed_package=fixed_package,
            fixed_runtime=fixed_runtime,
            standalone_package=standalone_package,
            standalone_runtime=standalone_runtime,
        )
        self.assertEqual(result["decision"], "SELECT_STANDALONE_DISTILLED_4B")
        self.assertEqual(
            result["fixed_replacement"]["status"],
            "DEPLOYABLE_EXPERIMENT_NOT_METRIC_PROMOTED",
        )


if __name__ == "__main__":
    unittest.main()
