from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from select_candidate import canonical_sha256
from standalone_run import load_config, open_image, rank_percentile, user_text
from build_package import main as build_package


class RuntimeContractTest(unittest.TestCase):
    def test_rank_calibration_averages_ties(self) -> None:
        np.testing.assert_array_equal(
            rank_percentile(np.asarray([2.0, 1.0, 2.0])),
            np.asarray([5 / 6, 1 / 3, 5 / 6], dtype=np.float32),
        )

    def test_image_and_prompt_match_student_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "image.png"
            Image.new("RGB", (900, 1200), "white").save(path)
            image = open_image(str(path))
            try:
                self.assertEqual(image.size, (336, 448))
            finally:
                image.close()
        row = type(
            "Row",
            (),
            {"category": "БАД", "name": "Товар", "description": "Описание"},
        )()
        self.assertTrue(user_text(row).endswith("Ответь только одной цифрой: 1 или 0."))

    def test_runtime_config_is_self_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "config.json"
            value = {
                "schema_version": "exp718_runtime_config_v1",
                "architecture": "qwen35_only",
                "base_model": "Qwen/Qwen3.5-4B",
                "score": "category_batch_percentile_rank",
                "threshold_rule": "median_of_five_outer_train_thresholds",
                "thresholds": {"БАД": 0.5, "Легковоспламеняющиеся": 0.9},
                "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
                "teacher_required_at_inference": False,
            }
            value["contract_sha256"] = canonical_sha256(value)
            path.write_text(json.dumps(value))
            self.assertEqual(load_config(path)["architecture"], "qwen35_only")

    def test_package_builder_binds_authorized_selection_and_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            model_bytes = b"distilled-adapter"
            (adapter / "adapter_model.safetensors").write_bytes(model_bytes)
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "r": 16,
                        "lora_alpha": 32,
                        "use_rslora": True,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    }
                )
            )
            full = {
                "schema_version": "exp715_full_refit_v1",
                "experiment_id": "715",
                "source_experiment_id": "698",
                "selected_mode": "hardneg_candidate",
                "full_data": True,
                "technical_smoke": False,
                "teacher_required_at_inference": False,
                "submission_base_model": "Qwen/Qwen3.5-4B",
                "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
                "adapter_model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                "adapter_bytes": len(model_bytes),
            }
            full["contract_sha256"] = canonical_sha256(full)
            full_path = root / "full.json"
            full_path.write_text(json.dumps(full))
            full_sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
            selection = {
                "schema_version": "exp718_standalone_selection_v1",
                "experiment_id": "718",
                "authorized": True,
                "decision": "STANDALONE_4B_AUTHORIZED",
                "selected_mode": "hardneg_candidate",
                "delta_vs_incumbent_140": 0.002,
                "deployment_thresholds": {"БАД": 0.5, "Легковоспламеняющиеся": 0.9},
                "deployment_score": "category_batch_percentile_rank",
                "deployment_threshold_rule": "median_of_five_outer_train_thresholds",
                "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
                "gates": {"a": True, "b": True},
                "full_refit_contract_sha256": full_sha,
                "adapter_model_sha256": full["adapter_model_sha256"],
                "teacher_required_at_inference": False,
                "submission_base_model": "Qwen/Qwen3.5-4B",
            }
            selection["contract_sha256"] = canonical_sha256(selection)
            selection_path = root / "selection.json"
            selection_path.write_text(json.dumps(selection))
            run_path = root / "run.py"
            run_path.write_text("print('standalone')\n")
            output = root / "submit.zip"
            report = root / "report.json"
            argv = [
                "build_package.py",
                "--selection", str(selection_path),
                "--full-refit-contract", str(full_path),
                "--adapter-dir", str(adapter),
                "--run-source", str(run_path),
                "--output", str(output),
                "--report", str(report),
            ]
            with patch.object(sys, "argv", argv):
                build_package()
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read("adapter_qwen35/adapter_model.safetensors"), model_bytes)
                runtime = json.loads(archive.read("standalone_config.json"))
                self.assertEqual(runtime["architecture"], "qwen35_only")
            self.assertTrue(json.loads(report.read_text())["under_5_gib"])


if __name__ == "__main__":
    unittest.main()
