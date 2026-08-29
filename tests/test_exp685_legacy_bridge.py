from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "experiments/685_qwen35_4b_structured_rank_distillation/bridge_legacy_teacher.py"
)
SPEC = importlib.util.spec_from_file_location("exp685_legacy_bridge", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LegacyBridgeTests(unittest.TestCase):
    @staticmethod
    def make_source(root: Path, payload: bytes) -> Path:
        source = root / "source"
        source.mkdir()
        with zipfile.ZipFile(source / "teacher_outer_train_scores.zip", "w") as archive:
            archive.writestr("report.json", "{}\n")
            archive.writestr("teacher_scores.jsonl", payload)
        (source / "teacher_scores.jsonl").write_bytes(payload)
        (source / "report.json").write_text("{}\n", encoding="utf-8")
        (source / "delivery.json").write_text("{}\n", encoding="utf-8")
        return source

    def test_bridge_verifies_before_copy_and_self_hashes(self):
        payload = b'{"score": 1.0}\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, payload)
            inner_sha = MODULE.sha256_file(source / "teacher_outer_train_scores.zip")
            output = root / "output"
            result = MODULE.bridge(
                {0: source},
                {0: inner_sha},
                {0: hashlib.sha256(payload).hexdigest()},
                output,
            )
            body = dict(result)
            digest = body.pop("bridge_sha256")
            self.assertEqual(digest, MODULE.canonical_sha256(body))
            self.assertTrue((output / "fold0/teacher_outer_train_scores.zip").is_file())
            self.assertEqual(
                {path.relative_to(output / "fold0").as_posix() for path in (output / "fold0").iterdir()},
                {"teacher_outer_train_scores.zip"},
            )
            self.assertEqual(result["decision"], "ACCEPT_REMOTE_BRIDGE")

    def test_wrong_score_hash_writes_nothing(self):
        payload = b'{"score": 1.0}\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, payload)
            inner_sha = MODULE.sha256_file(source / "teacher_outer_train_scores.zip")
            output = root / "output"
            with self.assertRaises(ValueError):
                MODULE.bridge({0: source}, {0: inner_sha}, {0: "0" * 64}, output)
            self.assertFalse(output.exists())

    def test_appledouble_and_extra_inner_members_fail_closed(self):
        payload = b'{"score": 1.0}\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, payload)
            (source / "._teacher_scores.jsonl").write_bytes(b"metadata")
            with self.assertRaisesRegex(ValueError, "transport metadata"):
                MODULE.safe_files(source)
            (source / "._teacher_scores.jsonl").unlink()
            with zipfile.ZipFile(source / "teacher_outer_train_scores.zip", "a") as archive:
                archive.writestr("extra.json", "{}\n")
            with self.assertRaisesRegex(ValueError, "member set"):
                MODULE.teacher_score_payload(
                    MODULE.safe_files(source),
                    MODULE.sha256_file(source / "teacher_outer_train_scores.zip"),
                )


if __name__ == "__main__":
    unittest.main()
