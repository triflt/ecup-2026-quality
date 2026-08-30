from __future__ import annotations

import unittest

from select_candidate import (
    MATCHED_CONTROL_VERIFICATION,
    build_selection,
    canonical_sha256,
)


def fixtures(macro: float = 0.914) -> tuple[dict, dict, dict]:
    category = {
        "БАД": {
            "f1": 0.952,
            "deployment": {
                "score": "category_batch_percentile_rank",
                "threshold": 0.52,
                "threshold_rule": "median_of_five_outer_train_thresholds",
                "outer_train_thresholds": [0.51, 0.52, 0.53, 0.52, 0.5],
            },
        },
        "Легковоспламеняющиеся": {
            "f1": 2 * macro - 0.952,
            "deployment": {
                "score": "category_batch_percentile_rank",
                "threshold": 0.91,
                "threshold_rule": "median_of_five_outer_train_thresholds",
                "outer_train_thresholds": [0.9, 0.91, 0.92, 0.91, 0.89],
            },
        },
    }
    modes = {
        mode: {
            "nested_macro_f1": macro if mode == "hardneg_candidate" else macro - 0.001,
            "score_calibration": "category_and_outer_fold_percentile_average_ties",
            "categories": category,
        }
        for mode in ("gold_control", "hardneg_candidate", "rank_candidate")
    }
    evaluation = {
        "schema_version": "exp698_evaluation_v2",
        "matched_control_verification": MATCHED_CONTROL_VERIFICATION,
        "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
        "deployment_score": "category_batch_percentile_rank",
        "deployment_threshold_rule": "median_of_five_outer_train_thresholds",
        "modes": modes,
        "comparisons": {
            "hardneg_candidate": {"science_gate": True, "fold_wins": 4},
            "rank_candidate": {"science_gate": False, "fold_wins": 2},
        },
    }
    guard = {
        "schema_version": "exp698_connected_family_guard_v2",
        "promoted_candidates": ["hardneg_candidate"],
        "audits": {"hardneg_candidate": {"promotion_gate": True}},
        "input_sha256": {"evaluation": "e" * 64},
    }
    full = {
        "schema_version": "exp715_full_refit_v1",
        "experiment_id": "715",
        "source_experiment_id": "698",
        "selected_mode": "hardneg_candidate",
        "full_data": True,
        "technical_smoke": False,
        "teacher_required_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
        "adapter_model_sha256": "a" * 64,
        "winner_binding": {
            "evaluation_sha256": "e" * 64,
            "connected_guard_sha256": "g" * 64,
            "score_calibration": "category_and_outer_fold_percentile_average_ties",
            "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
        },
    }
    full["contract_sha256"] = canonical_sha256(full)
    return evaluation, guard, full


class SelectionTest(unittest.TestCase):
    def test_rejects_stale_evaluation_without_matched_control_evidence(self) -> None:
        evaluation, guard, full = fixtures()
        evaluation.pop("matched_control_verification")
        with self.assertRaisesRegex(ValueError, "matched-control verification mismatch"):
            build_selection(
                evaluation,
                guard,
                full,
                evaluation_sha256="e" * 64,
                guard_sha256="g" * 64,
                full_refit_sha256="f" * 64,
            )

    def test_authorizes_strong_dual_gate_student(self) -> None:
        evaluation, guard, full = fixtures()
        result = build_selection(
            evaluation,
            guard,
            full,
            evaluation_sha256="e" * 64,
            guard_sha256="g" * 64,
            full_refit_sha256="f" * 64,
        )
        self.assertTrue(result["authorized"])
        self.assertEqual(result["decision"], "STANDALONE_4B_AUTHORIZED")
        self.assertEqual(result["deployment_thresholds"]["БАД"], 0.52)

    def test_rejects_candidate_below_incumbent(self) -> None:
        evaluation, guard, full = fixtures(macro=0.91)
        result = build_selection(
            evaluation,
            guard,
            full,
            evaluation_sha256="e" * 64,
            guard_sha256="g" * 64,
            full_refit_sha256="f" * 64,
        )
        self.assertFalse(result["authorized"])
        self.assertFalse(result["gates"]["macro_delta_vs_incumbent_at_least_0_001"])

    def test_no_promoted_candidate_is_fail_closed_without_refit(self) -> None:
        evaluation, guard, _ = fixtures()
        guard["promoted_candidates"] = []
        result = build_selection(
            evaluation,
            guard,
            None,
            evaluation_sha256="e" * 64,
            guard_sha256="g" * 64,
            full_refit_sha256=None,
        )
        self.assertFalse(result["authorized"])
        self.assertEqual(result["decision"], "NO_DISTILLATION_CANDIDATE_PROMOTED")


if __name__ == "__main__":
    unittest.main()
