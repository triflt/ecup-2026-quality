from __future__ import annotations

import unittest

from build_f3_full_targets import consistent_score_map


class FastTargetTest(unittest.TestCase):
    def test_repeated_occurrences_require_identical_scores(self) -> None:
        self.assertEqual(
            consistent_score_map(
                [{"id": "1", "score": 2.0}, {"id": "1", "score": 2.0}]
            ),
            {"1": 2.0},
        )
        with self.assertRaisesRegex(ValueError, "inconsistent repeated"):
            consistent_score_map(
                [{"id": "1", "score": 2.0}, {"id": "1", "score": 3.0}]
            )


if __name__ == "__main__":
    unittest.main()
