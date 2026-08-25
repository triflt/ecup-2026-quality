from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "experiments/685_qwen35_4b_structured_rank_distillation/build_pair_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("exp685_pair_runtime", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(index: int, label: int, *, component: str | None = None) -> dict[str, object]:
    return {
        "global_index": index,
        "id": str(index),
        "category": MODULE.FLAMMABLE,
        "fold": 1,
        "occurrence_index": 0,
        "semantic_component": component or f"component-{index}",
        "label": label,
    }


class RankDistillationPairTest(unittest.TestCase):
    def test_tie_aware_ranks_use_average_rank(self) -> None:
        self.assertEqual(
            MODULE.tie_aware_ranks([3.0, 1.0, 1.0, 2.0]),
            [4.0, 1.5, 1.5, 3.0],
        )

    def test_pair_manifest_is_deterministic_and_balanced(self) -> None:
        rows = [row(0, 1), row(1, 1)] + [
            row(index, 0) for index in range(2, 34)
        ]
        scores = [1.5, 0.5] + [0.4 - index / 20 for index in range(32)]
        first, first_summary = MODULE.build_pair_records(
            rows, scores, fold=0, max_endpoint_count=8
        )
        second, second_summary = MODULE.build_pair_records(
            rows, scores, fold=0, max_endpoint_count=8
        )
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(len(first), 16)
        self.assertEqual(first_summary["teacher_hard_fraction"], 0.5)
        self.assertEqual(
            {pair["pair_kind"] for pair in first},
            {"teacher_hard_balanced", "hash_uniform_balanced"},
        )
        for pair in first:
            self.assertNotEqual(pair["positive_key"], pair["negative_key"])
            self.assertNotEqual(
                pair["positive_semantic_component"],
                pair["negative_semantic_component"],
            )
            self.assertLessEqual(MODULE.TARGET_MIN, pair["pair_target"])
            self.assertLessEqual(pair["pair_target"], MODULE.TARGET_MAX)

    def test_same_component_negative_is_excluded(self) -> None:
        rows = [row(0, 1, component="shared")]
        rows += [row(1, 0, component="shared")]
        rows += [row(index, 0) for index in range(2, 18)]
        scores = [0.0, -0.01] + [-(index + 1) / 10 for index in range(16)]
        pairs, _ = MODULE.build_pair_records(
            rows, scores, fold=0, max_endpoint_count=8
        )
        self.assertTrue(all(pair["negative_key"][0] != 1 for pair in pairs))

    def test_positive_item_cap_fails_closed(self) -> None:
        rows = [row(0, 1), row(1, 1)] + [
            row(index, 0) for index in range(2, 20)
        ]
        rows[1]["id"] = rows[0]["id"]
        scores = [1.0, 0.9] + [-float(index) for index in range(18)]
        with self.assertRaisesRegex(ValueError, "positive item alone exceeds"):
            MODULE.build_pair_records(rows, scores, fold=0, max_endpoint_count=8)

    def test_outer_validation_row_fails_closed(self) -> None:
        rows = [row(0, 1)] + [row(index, 0) for index in range(1, 10)]
        rows[0]["fold"] = 0
        with self.assertRaisesRegex(ValueError, "outer-validation"):
            MODULE.build_pair_records(
                rows,
                [float(index) for index in range(len(rows))],
                fold=0,
                max_endpoint_count=8,
            )


if __name__ == "__main__":
    unittest.main()
