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
    def test_registry_packet_is_bound_by_exact_ordered_identity(self):
        development = [
            {
                "id": "a",
                "category": MODULE.FLAMMABLE,
                "label": "1",
                "semantic_component": "x",
                "development_fold": "0",
            },
            {
                "id": "b",
                "category": "other",
                "label": "0",
                "semantic_component": "z",
                "development_fold": "2",
            },
            {
                "id": "c",
                "category": MODULE.FLAMMABLE,
                "label": "0",
                "semantic_component": "y",
                "development_fold": "0",
            },
        ]
        validation = [
            {"global_index": 0, "id": "a", "fold": 0, "category": MODULE.FLAMMABLE},
            {"global_index": 2, "id": "c", "fold": 0, "category": MODULE.FLAMMABLE},
        ]
        labels, components, packet_sha = MODULE.bind_registry_rows(
            validation, development, expected_selected_rows=2
        )
        np.testing.assert_array_equal(labels, np.asarray([1, 0], dtype=np.int8))
        self.assertEqual(components, ["x", "y"])
        self.assertEqual(len(packet_sha), 64)
        validation[1]["global_index"] = 1
        with self.assertRaisesRegex(ValueError, "differs from frozen registry"):
            MODULE.bind_registry_rows(
                validation, development, expected_selected_rows=2
            )

    def test_registry_duplicate_ids_are_rejected(self):
        development = [
            {
                "id": "a",
                "category": MODULE.FLAMMABLE,
                "label": "1",
                "semantic_component": "x",
                "development_fold": "0",
            }
        ]
        with self.assertRaisesRegex(ValueError, "duplicate development ID"):
            MODULE.bind_registry_rows([], development * 2, expected_selected_rows=0)

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
