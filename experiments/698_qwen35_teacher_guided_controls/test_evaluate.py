from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import (
    EXPECTED_OBJECTIVES,
    EXPECTED_TRAINING_CONTRACT,
    fold_category_percentile,
    load_mode,
    nested_report,
    verify_matched_controls,
)


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluateTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for fold in range(5):
            for category in ("БАД", "Легковоспламеняющиеся"):
                for label in (0, 1):
                    rows.append(
                        {
                            "id": f"{fold}-{category}-{label}",
                            "fold": fold,
                            "category": category,
                            "label": label,
                        }
                    )
        self.data = pd.DataFrame(rows)

    def test_nested_report_is_outer_thresholded(self) -> None:
        scores = np.asarray([0.9 if label else 0.1 for label in self.data["label"]])
        report, predictions = nested_report(self.data, scores)
        self.assertEqual(report["nested_macro_f1"], 1.0)
        self.assertEqual(report["categories"]["БАД"]["deployment"]["threshold"], 0.5)
        np.testing.assert_array_equal(predictions, self.data["label"].to_numpy())

    def test_fold_category_calibration_removes_adapter_scale_and_offset(self) -> None:
        base = np.asarray([0.9 if label else 0.1 for label in self.data["label"]])
        shifted = base.copy()
        for index, row in self.data.iterrows():
            category_offset = 1000.0 if row["category"] == "БАД" else -1000.0
            shifted[index] = (row["fold"] + 1) * base[index] + 100.0 * row["fold"] + category_offset
        expected = fold_category_percentile(self.data, base)
        actual = fold_category_percentile(self.data, shifted)
        np.testing.assert_array_equal(actual, expected)
        report, predictions = nested_report(self.data, actual)
        self.assertEqual(report["nested_macro_f1"], 1.0)
        self.assertEqual(report["categories"]["БАД"]["deployment"]["threshold"], 0.75)
        np.testing.assert_array_equal(predictions, self.data["label"].to_numpy())

    def test_fold_category_calibration_averages_ties(self) -> None:
        scores = np.ones(len(self.data), dtype=np.float32)
        calibrated = fold_category_percentile(self.data, scores)
        np.testing.assert_array_equal(calibrated, np.full(len(self.data), 0.75, dtype=np.float32))

    def test_load_mode_verifies_five_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            mode = "gold_control"
            for fold in range(5):
                directory = root / f"{mode}-f{fold}"
                directory.mkdir()
                local = self.data.loc[self.data["fold"] == fold]
                predictions = [
                    {
                        "id": row.id,
                        "fold": fold,
                        "mode": mode,
                        "score": 0.9 if row.label else 0.1,
                    }
                    for row in local.itertuples()
                ]
                prediction_path = directory / "predictions.jsonl"
                prediction_path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
                )
                adapter = directory / "adapter"
                adapter.mkdir()
                (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
                (adapter / "adapter_config.json").write_text("{}")
                contract = {
                    "schema_version": "exp698_fold_output_v2",
                    "experiment_id": "698",
                    "source_experiment_id": "697",
                    "fold": fold,
                    "mode": mode,
                    "technical_smoke": False,
                    "model": "Qwen/Qwen3.5-4B",
                    "submission_eligible_base_model": True,
                    "teacher_model_required_at_inference": False,
                    "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
                    "train_occurrences": 5390,
                    "full_train_occurrences": 5390,
                    "validation_rows": len(predictions),
                    "micro_batch": 2,
                    "gradient_accumulation": 8,
                    "effective_batch": 16,
                    "optimizer_steps": 337,
                    "training_contract": EXPECTED_TRAINING_CONTRACT,
                    "objective_contract": EXPECTED_OBJECTIVES[mode],
                    "teacher_binding": None,
                    "runtime_audit_sha256": "a" * 64,
                    "runtime_minutes": 1.0,
                    "peak_cuda_memory_bytes": 1,
                    "predictions_sha256": file_sha256(prediction_path),
                    "adapter_model_sha256": file_sha256(
                        adapter / "adapter_model.safetensors"
                    ),
                    "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
                }
                contract["contract_sha256"] = canonical_sha256(contract)
                (directory / "output_contract.json").write_text(json.dumps(contract))
            scores, bindings = load_mode(root, mode, self.data)
            self.assertEqual(len(scores), len(self.data))
            self.assertEqual(len(bindings), 5)

    def test_matched_control_verifier_rejects_runtime_or_target_drift(self) -> None:
        teacher_binding = {
            "teacher_target_contract_sha256": "1" * 64,
            "teacher_targets_sha256": "2" * 64,
            "adapter_output_contract_sha256": "3" * 64,
            "adapter_model_sha256": "4" * 64,
            "outer_safe_validation_excluded": True,
            "targets_are_in_sample_within_outer_train": True,
        }
        report = {"modes": {mode: {"fold_artifacts": {}} for mode in EXPECTED_OBJECTIVES}}
        for fold in range(5):
            for mode in EXPECTED_OBJECTIVES:
                report["modes"][mode]["fold_artifacts"][str(fold)] = {
                    "runtime_audit_sha256": "a" * 64,
                    "training_contract": EXPECTED_TRAINING_CONTRACT,
                    "teacher_binding": None if mode == "gold_control" else teacher_binding,
                }
        verify_matched_controls(report)
        report["modes"]["rank_candidate"]["fold_artifacts"]["2"][
            "runtime_audit_sha256"
        ] = "b" * 64
        with self.assertRaisesRegex(ValueError, "common-factor mismatch"):
            verify_matched_controls(report)
        report["modes"]["rank_candidate"]["fold_artifacts"]["2"][
            "runtime_audit_sha256"
        ] = "a" * 64
        report["modes"]["rank_candidate"]["fold_artifacts"]["2"][
            "teacher_binding"
        ] = {**teacher_binding, "teacher_targets_sha256": "9" * 64}
        with self.assertRaisesRegex(ValueError, "teacher-target mismatch"):
            verify_matched_controls(report)


if __name__ == "__main__":
    unittest.main()
