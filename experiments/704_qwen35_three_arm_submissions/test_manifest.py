from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_manifest import MODES, build_manifest


class ManifestTest(unittest.TestCase):
    def test_three_bound_arms_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            for mode in MODES:
                archive = root / f"{mode}-submit.zip"
                archive.write_bytes(f"zip-{mode}".encode())
                report = {
                    "schema_version": "exp716_submission_package_v1",
                    "selected_mode": mode,
                    "output_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "output_bytes": archive.stat().st_size,
                    "teacher_in_submission": False,
                    "base_model_in_submission": False,
                    "under_5_gib": True,
                    "zip_integrity": "PASS",
                }
                (root / f"{mode}-package.json").write_text(json.dumps(report))
            manifest = build_manifest(root, "provisional_fold3")
            self.assertEqual(set(manifest["arms"]), set(MODES))
            self.assertEqual(manifest["decision"], "THREE_ARM_SUBMISSIONS_READY")


if __name__ == "__main__":
    unittest.main()
