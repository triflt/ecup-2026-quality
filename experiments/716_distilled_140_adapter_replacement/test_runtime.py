from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from finalize_runtime import main as finalize_main
from runtime_smoke import output_contract, safe_zip


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeContractTest(unittest.TestCase):
    def test_output_contract_is_exact(self) -> None:
        valid = pd.DataFrame(
            {
                "id": ["a", "b"],
                "result": [
                    "<комментарий>Текст и изображения не подтверждают обязательную маркировку товара.<вердикт>бан",
                    "<комментарий>Текст и изображения подтверждают наличие самостоятельного горючего товара.<вердикт>не бан",
                ],
            }
        )
        self.assertEqual(output_contract(valid, {"a", "b"}), (True, True))
        duplicate = valid.copy()
        duplicate["id"] = ["a", "a"]
        self.assertEqual(output_contract(duplicate, {"a", "b"}), (False, True))
        reordered = valid[["result", "id"]]
        self.assertEqual(output_contract(reordered, {"a", "b"}), (False, False))

    def test_safe_zip_requires_runtime_members(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            archive_path = Path(raw) / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("run.py", "")
            with self.assertRaisesRegex(ValueError, "lacks required runtime members"):
                safe_zip(archive_path)

    def test_final_acceptance_binds_both_smokes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            submission = root / "submission.zip"
            submission.write_bytes(b"submission")
            submission_sha = digest(submission)
            package = root / "package.json"
            package.write_text(
                json.dumps(
                    {
                        "schema_version": "exp716_submission_package_v1",
                        "experiment_id": "716",
                        "output_sha256": submission_sha,
                        "under_5_gib": True,
                        "teacher_in_submission": False,
                        "base_model_in_submission": False,
                        "zip_integrity": "PASS",
                    }
                )
            )

            def write_smoke(path: Path, rows: int) -> None:
                payload = {
                    "schema_version": "exp716_runtime_smoke_v1",
                    "experiment_id": "716",
                    "rows": rows,
                    "submission_sha256": submission_sha,
                    "return_code": 0,
                    "cuda_visible_devices": "0",
                    "output_schema_valid": True,
                    "unique_ids_valid": True,
                    "decision": "RUNTIME_SMOKE_PASS",
                    "peak_gpu_memory_mib": 1234,
                    "elapsed_seconds": 60.0,
                }
                if rows == 600:
                    payload["projected_public_minutes_1600"] = 2.67
                    payload["projected_private_minutes_3800"] = 6.33
                path.write_text(json.dumps(payload))

            smoke3, smoke600 = root / "smoke3.json", root / "smoke600.json"
            write_smoke(smoke3, 3)
            write_smoke(smoke600, 600)
            output = root / "acceptance.json"
            argv = [
                "finalize_runtime.py",
                "--submission", str(submission),
                "--package-report", str(package),
                "--smoke-3", str(smoke3),
                "--smoke-600", str(smoke600),
                "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                finalize_main()
            result = json.loads(output.read_text())
            self.assertEqual(result["decision"], "DEPLOYABLE_RUNTIME_PASS")
            self.assertEqual(result["measured_rows"], 600)
            self.assertEqual(result["submission_sha256"], submission_sha)


if __name__ == "__main__":
    unittest.main()
