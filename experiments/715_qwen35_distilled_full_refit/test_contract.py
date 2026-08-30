from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_full_runtime import sampled_indices
from run_full import (
    MATCHED_CONTROL_VERIFICATION,
    resolve_explicit_mode,
    resolve_winner,
)


class FullRefitContractTest(unittest.TestCase):
    def test_full_sampler_is_deterministic_and_oversamples_flammable(self) -> None:
        rows = []
        for category, label, count in (
            ("БАД", 0, 5),
            ("БАД", 1, 5),
            ("Легковоспламеняющиеся", 0, 6),
            ("Легковоспламеняющиеся", 1, 2),
        ):
            rows.extend({"category": category, "label": label} for _ in range(count))
        frame = pd.DataFrame(rows)
        first, audit = sampled_indices(frame)
        second, _ = sampled_indices(frame)
        self.assertEqual(first, second)
        self.assertEqual(audit["flammable_positive_occurrences"], 10)
        self.assertGreater(len(first), len(set(first)))

    def test_winner_requires_guard_binding_and_uses_primary_delta(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            evaluation_path = root / "evaluation.json"
            guard_path = root / "guard.json"
            evaluation = {
                "schema_version": "exp698_evaluation_v2",
                "matched_control_verification": MATCHED_CONTROL_VERIFICATION,
                "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
                "modes": {
                    mode: {
                        "score_calibration": "category_and_outer_fold_percentile_average_ties"
                    }
                    for mode in ("gold_control", "hardneg_candidate", "rank_candidate")
                },
                "comparisons": {
                    "hardneg_candidate": {"nested_macro_delta_vs_gold_control": 0.002},
                    "rank_candidate": {"nested_macro_delta_vs_gold_control": 0.003},
                },
            }
            evaluation_path.write_text(json.dumps(evaluation))
            evaluation_sha = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
            guard = {
                "schema_version": "exp698_connected_family_guard_v2",
                "promoted_candidates": ["hardneg_candidate", "rank_candidate"],
                "input_sha256": {"evaluation": evaluation_sha},
            }
            guard_path.write_text(json.dumps(guard))
            winner, binding = resolve_winner(evaluation_path, guard_path)
            self.assertEqual(winner, "rank_candidate")
            self.assertEqual(binding["evaluation_sha256"], evaluation_sha)
            evaluation.pop("matched_control_verification")
            evaluation_path.write_text(json.dumps(evaluation))
            with self.assertRaisesRegex(ValueError, "matched-control verification mismatch"):
                resolve_winner(evaluation_path, guard_path)

    def test_explicit_control_is_bound_but_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            evaluation_path = root / "evaluation.json"
            guard_path = root / "guard.json"
            evaluation = {
                "schema_version": "exp698_evaluation_v2",
                "matched_control_verification": MATCHED_CONTROL_VERIFICATION,
                "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
                "modes": {
                    mode: {
                        "score_calibration": "category_and_outer_fold_percentile_average_ties"
                    }
                    for mode in ("gold_control", "hardneg_candidate", "rank_candidate")
                },
                "comparisons": {
                    "hardneg_candidate": {"nested_macro_delta_vs_gold_control": 0.0},
                    "rank_candidate": {"nested_macro_delta_vs_gold_control": -0.001},
                },
            }
            evaluation_path.write_text(json.dumps(evaluation))
            guard = {
                "schema_version": "exp698_connected_family_guard_v2",
                "promoted_candidates": [],
                "input_sha256": {
                    "evaluation": hashlib.sha256(
                        evaluation_path.read_bytes()
                    ).hexdigest()
                },
            }
            guard_path.write_text(json.dumps(guard))
            mode, binding = resolve_explicit_mode(
                evaluation_path, guard_path, "gold_control"
            )
            self.assertEqual(mode, "gold_control")
            self.assertEqual(binding["promoted_candidates"], [])
            self.assertEqual(
                binding["selection_rule"], "explicit_three_arm_auxiliary_refit"
            )


if __name__ == "__main__":
    unittest.main()
