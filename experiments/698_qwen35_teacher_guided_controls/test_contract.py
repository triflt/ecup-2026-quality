from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from run_fold import (
    TEACHER_MODEL,
    common_training_contract,
    hard_example_weight,
    load_teacher_targets,
    objective_contract,
    open_image,
    smoke_indices,
    within_stratum_rank_loss,
)


class ObjectiveTest(unittest.TestCase):
    def test_only_objective_contract_differs_across_matched_arms(self) -> None:
        order_sha = "a" * 64
        self.assertEqual(
            common_training_contract(order_sha), common_training_contract(order_sha)
        )
        objectives = {
            mode: objective_contract(mode)
            for mode in ("gold_control", "hardneg_candidate", "rank_candidate")
        }
        self.assertEqual(len({json.dumps(value, sort_keys=True) for value in objectives.values()}), 3)
        self.assertFalse(objectives["gold_control"]["teacher_target_consumed"])
        self.assertTrue(objectives["hardneg_candidate"]["teacher_target_consumed"])
        self.assertTrue(objectives["rank_candidate"]["teacher_target_consumed"])

    def test_image_preprocessing_matches_solution_140_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "image.png"
            Image.new("RGB", (900, 1200), "white").save(path)
            image = open_image({"image_path": str(path)})
            try:
                self.assertEqual(image.size, (336, 448))
            finally:
                image.close()

    def test_hard_weight_only_changes_teacher_disagreement(self) -> None:
        row = {"category": "Легковоспламеняющиеся", "label": 1}
        self.assertEqual(hard_example_weight(row, 2.0), 1.0)
        self.assertGreater(hard_example_weight(row, -2.0), 1.0)
        bad = {"category": "БАД", "label": 1}
        self.assertEqual(hard_example_weight(bad, -3.0), 1.0)

    def test_rank_is_same_label_flammable_only(self) -> None:
        rows = [
            {"category": "Легковоспламеняющиеся", "label": 1},
            {"category": "Легковоспламеняющиеся", "label": 1},
            {"category": "Легковоспламеняющиеся", "label": 0},
            {"category": "БАД", "label": 1},
        ]
        loss, pairs = within_stratum_rank_loss(
            torch.tensor([0.0, 1.0, -1.0, 0.5]), rows, [2.0, 1.0, -2.0, 3.0]
        )
        self.assertEqual(pairs, 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_smoke_selection_exercises_changed_factor(self) -> None:
        rows = [
            {"category": "БАД", "label": 0},
            {"category": "Легковоспламеняющиеся", "label": 1},
            {"category": "Легковоспламеняющиеся", "label": 1},
            {"category": "Легковоспламеняющиеся", "label": 0},
        ]
        teacher = [0.0, -2.0, 1.0, -1.0]
        hard = smoke_indices("hardneg_candidate", rows, teacher)
        self.assertEqual(hard[0], 1)
        rank = smoke_indices("rank_candidate", rows, teacher)
        self.assertEqual(rank[:2], [1, 2])

    def test_teacher_targets_bind_exact_runtime_occurrences(self) -> None:
        rows = [
            {"id": "a", "fold": 1, "category": "БАД", "label": 0},
            {
                "id": "b",
                "fold": 2,
                "category": "Легковоспламеняющиеся",
                "label": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            targets = [
                {
                    "occurrence_index": index,
                    "id": row["id"],
                    "outer_fold": 0,
                    "source_fold": row["fold"],
                    "category": row["category"],
                    "label": row["label"],
                    "score": float(index),
                }
                for index, row in enumerate(rows)
            ]
            target_path = root / "teacher_targets.jsonl"
            target_path.write_text("".join(json.dumps(row) + "\n" for row in targets))
            target_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
            contract = {
                "schema_version": "exp697_teacher_targets_v1",
                "experiment_id": "697",
                "fold": 0,
                "model": str(TEACHER_MODEL),
                "outer_safe_validation_excluded": True,
                "target_scope": "outer_train_occurrences",
                "targets_are_in_sample_within_outer_train": True,
                "teacher_frozen_before_target_generation": True,
                "atomic_directory_publish": True,
                "rows": 2,
                "unique_ids": 2,
                "inference_scope": "unique_consumed_flammable_ids_only",
                "teacher_signal_consumed_categories": [
                    "Легковоспламеняющиеся"
                ],
                "neutral_score_for_unconsumed_categories": 0.0,
                "inference_occurrences": 1,
                "inference_unique_ids": 1,
                "deduplicated_by_id": True,
                "runtime_audit_sha256": "a" * 64,
                "train_runtime_sha256": "b" * 64,
                "teacher_targets_sha256": target_sha,
                "adapter_model_sha256": "c" * 64,
                "adapter_output_contract_sha256": "d" * 64,
            }
            (root / "teacher_target_contract.json").write_text(json.dumps(contract))
            values, binding = load_teacher_targets(
                root,
                rows,
                0,
                runtime_audit_sha256="a" * 64,
                train_runtime_sha256="b" * 64,
            )
            self.assertEqual(values, [0.0, 1.0])
            self.assertEqual(binding["teacher_targets_sha256"], target_sha)


if __name__ == "__main__":
    unittest.main()
