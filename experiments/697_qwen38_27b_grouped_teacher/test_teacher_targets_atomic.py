from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from atomic_publish import atomic_output_directory
from generate_teacher_targets import FLAMMABLE, target_records, unique_consumed_rows


class AtomicTeacherTargetsTest(unittest.TestCase):
    def test_only_consumed_flammable_ids_are_inferred_and_expanded(self) -> None:
        rows = [
            {
                "id": "bad",
                "image_path": "/bad.jpg",
                "category": "БАД",
                "label": 1,
                "fold": 1,
            },
            {
                "id": "fire",
                "image_path": "/fire.jpg",
                "category": FLAMMABLE,
                "label": 1,
                "fold": 2,
            },
            {
                "id": "fire",
                "image_path": "/fire.jpg",
                "category": FLAMMABLE,
                "label": 1,
                "fold": 2,
            },
        ]
        selected = unique_consumed_rows(rows)
        self.assertEqual([row["id"] for row in selected], ["fire"])
        records = target_records(rows, 0, {"fire": 2.5})
        self.assertEqual([row["score"] for row in records], [0.0, 2.5, 2.5])
        self.assertEqual([row["occurrence_index"] for row in records], [0, 1, 2])

    def test_conflicting_repeated_id_is_rejected(self) -> None:
        rows = [
            {
                "id": "fire",
                "image_path": "/a.jpg",
                "category": FLAMMABLE,
                "label": 1,
                "fold": 2,
            },
            {
                "id": "fire",
                "image_path": "/b.jpg",
                "category": FLAMMABLE,
                "label": 1,
                "fold": 2,
            },
        ]
        with self.assertRaisesRegex(ValueError, "conflicting identity"):
            unique_consumed_rows(rows)

    def test_success_publishes_complete_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            output = Path(raw_tmp) / "teacher-targets-f0"
            with atomic_output_directory(output) as staging:
                (staging / "teacher_targets.jsonl").write_text("{}\n")
                (staging / "teacher_target_contract.json").write_text("{}\n")
                self.assertFalse(output.exists())
            self.assertTrue((output / "teacher_targets.jsonl").is_file())
            self.assertTrue((output / "teacher_target_contract.json").is_file())
            self.assertEqual(list(output.parent.glob(".teacher-targets-f0.staging-*")), [])

    def test_failure_leaves_no_final_or_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            output = Path(raw_tmp) / "teacher-targets-f0"
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with atomic_output_directory(output) as staging:
                    (staging / "teacher_targets.jsonl").write_text("partial\n")
                    raise RuntimeError("synthetic failure")
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(".teacher-targets-f0.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
