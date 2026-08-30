from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from build_submission import canonical_sha256, main, safe_member, verify_full_refit


class PackageTest(unittest.TestCase):
    def test_paths_fail_closed(self) -> None:
        self.assertTrue(safe_member("adapter_qwen35/adapter_config.json"))
        self.assertFalse(safe_member("../adapter_model.safetensors"))
        self.assertFalse(safe_member("/absolute/run.py"))
        self.assertFalse(safe_member("bad\\path"))
        self.assertFalse(PurePosixPath("../x").is_absolute())

    def test_canonical_hash_ignores_mapping_insertion_order(self) -> None:
        left = {"a": "1", "b": "2"}
        right = {"b": "2", "a": "1"}
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))
        self.assertEqual(json.loads(json.dumps(left)), left)

    def test_fold_control_adapter_requires_explicit_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            weights = b"provisional-control"
            config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "use_rslora": True,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
            (adapter / "adapter_model.safetensors").write_bytes(weights)
            (adapter / "adapter_config.json").write_text(json.dumps(config))
            contract = {
                "schema_version": "exp698_fold_output_v2",
                "experiment_id": "698",
                "source_experiment_id": "697",
                "fold": 3,
                "mode": "gold_control",
                "technical_smoke": False,
                "submission_eligible_base_model": True,
                "teacher_model_required_at_inference": False,
                "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
                "adapter_model_sha256": hashlib.sha256(weights).hexdigest(),
                "adapter_config_sha256": hashlib.sha256(
                    (adapter / "adapter_config.json").read_bytes()
                ).hexdigest(),
            }
            contract["contract_sha256"] = canonical_sha256(contract)
            path = root / "output_contract.json"
            path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "not authorized"):
                verify_full_refit(adapter, path)
            normalized, _ = verify_full_refit(
                adapter,
                path,
                allow_fold_contract=True,
                allow_control_mode=True,
            )
            self.assertEqual(normalized["selected_mode"], "gold_control")
            self.assertEqual(normalized["adapter_scope"], "outer_fold")

    def test_end_to_end_replaces_only_qwen35_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            base = root / "base.zip"
            fixed = {
                "run.py": b"print('fixed')\n",
                "adapter_qwen3vl/adapter_config.json": b"{}",
                "adapter_qwen3vl/adapter_model.safetensors": b"qwen3vl",
                "fixed.bin": b"fixed-component",
            }
            with zipfile.ZipFile(base, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, payload in fixed.items():
                    archive.writestr(name, payload)
                archive.writestr("metadata.json", json.dumps({"description": "base"}))
                archive.writestr("adapter_qwen35/adapter_config.json", b"{}")
                archive.writestr("adapter_qwen35/adapter_model.safetensors", b"old")
            adapter = root / "adapter"
            adapter.mkdir()
            adapter_payload = b"new-distilled-adapter"
            (adapter / "adapter_model.safetensors").write_bytes(adapter_payload)
            config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "use_rslora": True,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
            (adapter / "adapter_config.json").write_text(json.dumps(config))
            contract = {
                "schema_version": "exp715_full_refit_v1",
                "experiment_id": "715",
                "source_experiment_id": "698",
                "selected_mode": "hardneg_candidate",
                "full_data": True,
                "technical_smoke": False,
                "teacher_required_at_inference": False,
                "submission_base_model": "Qwen/Qwen3.5-4B",
                "image_preprocessing": "solution140_first_image_thumbnail_448_lanczos_v1",
                "adapter_model_sha256": hashlib.sha256(adapter_payload).hexdigest(),
                "adapter_bytes": len(adapter_payload),
            }
            contract["contract_sha256"] = canonical_sha256(contract)
            contract_path = root / "output_contract.json"
            contract_path.write_text(json.dumps(contract))
            evidence = {
                "schema_version": "exp717_incumbent_evidence_audit_v1",
                "experiment_id": "717",
                "exact_component_oof_available": False,
                "decision": "FREEZE_FUSION_USE_ISOLATED_REPLACEMENT_GATES",
                "incumbent_140": {"nested_macro_f1": 0.9118425205786493},
                "evidence_capabilities": {
                    "measure_exact_fixed_140_fusion_delta": False,
                    "retune_fusion_without_exact_component_oof": False,
                    "build_isolated_qwen35_adapter_replacement": True,
                },
            }
            evidence["contract_sha256"] = canonical_sha256(evidence)
            evidence_path = root / "incumbent_evidence_audit.json"
            evidence_path.write_text(json.dumps(evidence))
            output = root / "result.zip"
            report = root / "report.json"
            argv = [
                "build_submission.py",
                "--base-submission",
                str(base),
                "--expected-base-sha256",
                hashlib.sha256(base.read_bytes()).hexdigest(),
                "--adapter-dir",
                str(adapter),
                "--full-refit-contract",
                str(contract_path),
                "--incumbent-evidence-audit",
                str(evidence_path),
                "--output",
                str(output),
                "--report",
                str(report),
            ]
            with patch.object(sys, "argv", argv):
                main()
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read("fixed.bin"), fixed["fixed.bin"])
                self.assertEqual(
                    archive.read("adapter_qwen35/adapter_model.safetensors"),
                    adapter_payload,
                )
                distillation = json.loads(archive.read("metadata.json"))["distillation"]
                self.assertEqual(
                    distillation["fusion_policy"],
                    "FREEZE_FUSION_USE_ISOLATED_REPLACEMENT_GATES",
                )
                self.assertEqual(
                    distillation["image_preprocessing"],
                    "solution140_first_image_thumbnail_448_lanczos_v1",
                )
            self.assertTrue(json.loads(report.read_text())["under_5_gib"])


if __name__ == "__main__":
    unittest.main()
