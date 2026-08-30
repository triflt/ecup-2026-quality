from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from assemble_oof import assemble


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AssembleOofTest(unittest.TestCase):
    def test_strict_five_fold_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            experiment = root / "experiment"
            rows = []
            folds = []
            for fold in range(5):
                for category in ("BAD", "flammable"):
                    for label in (0, 1):
                        row_id = f"{fold}-{category}-{label}"
                        rows.append({"id": row_id, "category": category, "label": label})
                        folds.append({"id": row_id, "fold": fold})
            data_path = root / "data.csv"
            folds_path = root / "folds.csv"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            pd.DataFrame(folds).to_csv(folds_path, index=False)

            for fold in range(5):
                output = experiment / ".local" / f"output-f{fold}"
                output.mkdir(parents=True)
                predictions = [
                    {
                        "id": row["id"],
                        "fold": fold,
                        # Raw score offsets differ drastically across adapters.
                        # Only within-fold ordering is intentionally stable.
                        "score": fold * 100.0 + (0.9 if row["label"] else 0.1),
                    }
                    for row in rows
                    if row["id"].startswith(f"{fold}-")
                ]
                prediction_path = output / "predictions.jsonl"
                prediction_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in predictions)
                )
                contract = {
                    "experiment_id": "697",
                    "fold": fold,
                    "technical_smoke": False,
                    "validation_rows": len(predictions),
                    "predictions_sha256": file_sha256(prediction_path),
                }
                (output / "output_contract.json").write_text(json.dumps(contract))

            report = assemble(
                data_path=data_path,
                folds_path=folds_path,
                experiment_dir=experiment,
                output_dir=root / "oof",
                expected_data_sha256=file_sha256(data_path),
            )
            self.assertEqual(report["rows"], 20)
            self.assertEqual(report["schema_version"], "exp697_teacher_oof_v2")
            self.assertEqual(
                report["score_calibration"],
                "label_blind_percentile_rank_within_fold_and_category_average_ties",
            )
            self.assertEqual(report["nested_macro_f1"], 1.0)
            oof = pd.read_csv(root / "oof" / "teacher_oof.csv")
            self.assertEqual(len(oof), 20)
            self.assertEqual(set(oof["teacher_rank"]), {0.5, 1.0})


if __name__ == "__main__":
    unittest.main()
