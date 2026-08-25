from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/685_qwen35_4b_structured_rank_distillation"
sys.path.insert(0, str(EXP))
SPEC = importlib.util.spec_from_file_location("exp685_fast_eval", EXP / "evaluate_outer0_fast.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FastOuter0EvalTests(unittest.TestCase):
    def test_label_donor_is_bound_by_exact_global_index_and_identity(self):
        validation = [
            {"global_index": 10, "id": "a", "fold": 0, "category": MODULE.FLAMMABLE},
            {"global_index": 11, "id": "b", "fold": 0, "category": MODULE.FLAMMABLE},
        ]
        donor = [
            {**validation[0], "label": 1, "semantic_component": "x"},
            {**validation[1], "label": 0, "semantic_component": "y"},
        ]
        labels, components = MODULE.bind_outer0_labels(validation, donor)
        np.testing.assert_array_equal(labels, np.asarray([1, 0], dtype=np.int8))
        self.assertEqual(components, ["x", "y"])
        donor[1]["id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            MODULE.bind_outer0_labels(validation, donor)

    def test_consistent_donor_occurrence_duplicates_are_collapsed(self):
        validation = [
            {"global_index": 10, "id": "a", "fold": 0, "category": MODULE.FLAMMABLE}
        ]
        original = {
            **validation[0],
            "label": 1,
            "semantic_component": "x",
            "occurrence_index": 0,
        }
        repeated = {**original, "occurrence_index": 7}
        labels, components = MODULE.bind_outer0_labels(
            validation, [original, repeated]
        )
        np.testing.assert_array_equal(labels, np.asarray([1], dtype=np.int8))
        self.assertEqual(components, ["x"])
        conflicting = {**repeated, "label": 0}
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            MODULE.bind_outer0_labels(validation, [original, conflicting])

    def test_positive_class_metrics_and_correction_gate(self):
        labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        control = np.asarray([1, 0, 1, 0], dtype=np.int8)
        candidate = np.asarray([1, 1, 0, 0], dtype=np.int8)
        metrics = MODULE.positive_class_metrics(labels, candidate)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["f1"], 1.0)
        corrected = int(((control != labels) & (candidate == labels)).sum())
        regressed = int(((control == labels) & (candidate != labels)).sum())
        self.assertEqual((corrected, regressed), (2, 0))
        self.assertTrue(MODULE.correction_ratio_ok(corrected, regressed, 1.2))
        self.assertFalse(MODULE.correction_ratio_ok(0, 0, 1.2))


if __name__ == "__main__":
    unittest.main()
