from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "experiments/685_qwen35_4b_structured_rank_distillation/build_remote_compute_preset.py"
)
SPEC = importlib.util.spec_from_file_location("exp685_remote_compute_preset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


BASE = """job:
  time_limit: 2h0m0s
  flavor: gpu-h100-1-80
  region: msk
  image: example/image:immutable
  preemption: never
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: false
    PYTORCH_ALLOC_CONF: expandable_segments:True
"""


class RemotePresetTests(unittest.TestCase):
    def test_path_scope_is_fail_closed(self):
        self.assertEqual(
            MODULE.safe_s3_path(
                "/approved/project/exp685/code/abc",
                "/approved/project",
            ),
            "/approved/project/exp685/code/abc",
        )
        for unsafe in (
            "/other/path",
            "/approved/project/../secret",
            "/approved/project/has space",
        ):
            with self.assertRaises(ValueError):
                MODULE.safe_s3_path(unsafe, "/approved/project")

    def test_training_output_is_s3_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.yaml"
            base.write_text(BASE, encoding="utf-8")
            model = root / "model.txt"
            model.write_text(
                "type: model_registry, name: qwen, dst: /hf_models/",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                base_preset=base,
                time_limit=None,
                flavor=None,
                bucket="approved-bucket",
                code_src="/approved/project/exp685/code/abc",
                source_src="/approved/project/exp685/source/f0/abc",
                pair_src="/approved/project/exp685/pairs/f0/abc",
                images_src="/approved/project/images/frozen",
                output_dst="/approved/project/exp685/train/f0/control/run1",
                model_input_line_file=model,
                fold=0,
                mode="paired_hard_control",
                technical_smoke=False,
            )
            payload = MODULE.build_train(args)
        self.assertNotIn("type: files", payload)
        self.assertEqual(payload.count("type: s3msk"), 5)
        self.assertIn("when: on_job_status=succeeded", payload)
        self.assertIn("location: cluster", payload)
        self.assertIn("verify_training_artifact.py", payload)
        self.assertNotIn("adapter.zip", payload)


if __name__ == "__main__":
    unittest.main()
