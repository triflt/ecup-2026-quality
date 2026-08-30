from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import build_oof_targets as target_builder


class BuildOofTargetsTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        runtime = root / "runtime"
        runtime.mkdir()
        teacher_rows = []
        for index in range(target_builder.EXPECTED_TEACHER_ROWS):
            teacher_rows.append(
                {
                    "id": f"row-{index}",
                    "fold": index % 5,
                    "category": "BAD" if index % 2 == 0 else "flammable",
                    "label": index % 2,
                    "teacher_score": float(index % 37),
                    "teacher_prediction": index % 2,
                }
            )
        teacher = pd.DataFrame(teacher_rows)
        teacher["teacher_rank"] = teacher.groupby(
            ["fold", "category"], sort=False
        )["teacher_score"].rank(method="average", pct=True)
        teacher = teacher[
            [
                "id",
                "fold",
                "category",
                "label",
                "teacher_score",
                "teacher_rank",
                "teacher_prediction",
            ]
        ]
        teacher_path = root / "teacher_oof.csv"
        teacher.to_csv(teacher_path, index=False)

        train_rows = [
            {
                "id": row["id"],
                "fold": row["fold"],
                "category": row["category"],
                "label": row["label"],
            }
            for row in teacher_rows[:12]
        ]
        train_path = runtime / "train.jsonl"
        train_path.write_text(
            "".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8"
        )
        audit = {
            "schema_version": "exp715_full_runtime_v1",
            "train_sha256": target_builder.sha256(train_path),
        }
        (runtime / "runtime_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        report = {
            "schema_version": "exp697_teacher_oof_v2",
            "experiment_id": "697",
            "evaluation": "nested_grouped_fold_category_percentile_rank_v2",
            "score_calibration": target_builder.EXPECTED_SCORE_CALIBRATION,
            "raw_teacher_score_preserved": True,
            "rows": target_builder.EXPECTED_TEACHER_ROWS,
            "unique_ids": target_builder.EXPECTED_TEACHER_ROWS,
            "data_sha256": target_builder.EXPECTED_DATA_SHA256,
            "folds_sha256": target_builder.EXPECTED_FOLDS_SHA256,
            "teacher_oof_sha256": target_builder.sha256(teacher_path),
        }
        report_path = root / "teacher_oof_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return runtime, teacher_path, report_path

    def run_builder(
        self, runtime: Path, teacher_path: Path, report_path: Path, output: Path
    ) -> None:
        argv = [
            "build_oof_targets.py",
            "--runtime-dir",
            str(runtime),
            "--teacher-oof",
            str(teacher_path),
            "--teacher-report",
            str(report_path),
            "--output-dir",
            str(output),
        ]
        with patch.object(sys, "argv", argv):
            with redirect_stdout(io.StringIO()):
                target_builder.main()

    def test_builds_targets_from_verified_v2_rank(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            runtime, teacher_path, report_path = self.make_fixture(root)
            output = root / "targets"
            self.run_builder(runtime, teacher_path, report_path, output)
            contract = json.loads((output / "teacher_target_contract.json").read_text())
            self.assertEqual(contract["rows"], 12)
            self.assertEqual(
                contract["rank_score_calibration"],
                "category_and_source_fold_percentile_average_ties",
            )

    def test_rejects_rank_not_derived_from_raw_fold_category_scores(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            runtime, teacher_path, report_path = self.make_fixture(root)
            teacher = pd.read_csv(teacher_path)
            teacher.loc[0, "teacher_rank"] = 0.123456
            teacher.to_csv(teacher_path, index=False)
            report = json.loads(report_path.read_text())
            report["teacher_oof_sha256"] = target_builder.sha256(teacher_path)
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "rank calibration mismatch"):
                self.run_builder(runtime, teacher_path, report_path, root / "targets")


if __name__ == "__main__":
    unittest.main()
