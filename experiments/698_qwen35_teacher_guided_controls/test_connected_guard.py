from __future__ import annotations

import unittest

import numpy as np

from connected_guard import MATCHED_CONTROL_VERIFICATION, audit_candidate


class ConnectedGuardTest(unittest.TestCase):
    def test_matched_control_marker_is_versioned(self) -> None:
        self.assertEqual(
            MATCHED_CONTROL_VERIFICATION["schema_version"],
            "exp698_matched_control_verification_v1",
        )
        self.assertTrue(
            MATCHED_CONTROL_VERIFICATION["common_training_factors_equal_per_fold"]
        )

    def test_component_bootstrap_and_gate(self) -> None:
        labels = []
        categories = []
        folds = []
        components = []
        for fold in range(5):
            for category in ("БАД", "Легковоспламеняющиеся"):
                for label in (0, 1):
                    labels.append(label)
                    categories.append(category)
                    folds.append(fold)
                    components.append(f"{fold}-{category}-{label}")
        labels_array = np.asarray(labels, dtype=np.int8)
        result = audit_candidate(
            labels=labels_array,
            categories=np.asarray(categories),
            folds=np.asarray(folds, dtype=np.int8),
            components=np.asarray(components),
            safe=np.ones(len(labels), dtype=bool),
            control=1 - labels_array,
            candidate=labels_array.copy(),
            bootstrap=100,
            seed=42,
        )
        self.assertEqual(result["fold_wins"], 5)
        self.assertEqual(
            result["component_bootstrap"]["probability_delta_positive"], 1.0
        )
        self.assertTrue(result["objective_semantic_gate"])


if __name__ == "__main__":
    unittest.main()
