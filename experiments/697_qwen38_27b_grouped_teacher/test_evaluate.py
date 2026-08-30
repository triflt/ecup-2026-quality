from __future__ import annotations

import unittest

import pandas as pd

from evaluate import evaluate_predictions


class EvaluateTest(unittest.TestCase):
    def fixtures(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = []
        folds = []
        predictions = []
        for fold in (0, 3):
            for category in ("BAD", "flammable"):
                for label in (0, 1):
                    row_id = f"{fold}-{category}-{label}"
                    data.append({"id": row_id, "category": category, "label": label})
                    folds.append({"id": row_id, "fold": fold})
                    predictions.append(
                        {
                            "id": row_id,
                            "fold": fold,
                            # Fold 3 has a disjoint raw-logit scale.
                            "score": fold * 100.0 + label,
                        }
                    )
        return pd.DataFrame(predictions), pd.DataFrame(data), pd.DataFrame(folds)

    def test_screen_uses_fold_category_rank_not_cross_adapter_raw_logits(self) -> None:
        pred, data, folds = self.fixtures()
        result = evaluate_predictions(pred, data, folds)
        self.assertEqual(result["schema_version"], "exp697_screen_evaluation_v2")
        self.assertEqual(result["nested_macro_f1"], 1.0)
        self.assertEqual(result["decision_scope"], "screen_only_not_comparable_to_full_140")

    def test_rejects_incomplete_fold_coverage(self) -> None:
        pred, data, folds = self.fixtures()
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            evaluate_predictions(pred.iloc[:-1], data, folds)


if __name__ == "__main__":
    unittest.main()
