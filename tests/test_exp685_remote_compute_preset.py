from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/685_qwen35_4b_structured_rank_distillation/build_remote_compute_preset.py"
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
    def test_vault_s3_auth_is_all_or_nothing_and_never_literal(self):
        empty = argparse.Namespace()
        self.assertEqual(MODULE.s3_auth(empty), (None, None, None))
        incomplete = argparse.Namespace(
            vault_auth_role="team__ds_data_plane-ro",
            s3_access_key_vault_ref="vault:team/data/s3#access_key",
            s3_secret_key_vault_ref=None,
        )
        with self.assertRaises(ValueError):
            MODULE.s3_auth(incomplete)

        configured = argparse.Namespace(
            vault_auth_role="team__ds_data_plane-ro",
            s3_access_key_vault_ref="vault:team/data/s3#access_key",
            s3_secret_key_vault_ref="vault:team/data/s3#secret_key",
        )
        self.assertEqual(
            MODULE.s3_auth(configured),
            (
                "team__ds_data_plane-ro",
                "vault:team/data/s3#access_key",
                "vault:team/data/s3#secret_key",
            ),
        )

    def test_direct_s3_credentials_are_loaded_only_from_explicit_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env.s3"
            env_file.write_text(
                "TEST_ACCESS_KEY=synthetic-access\n"
                "TEST_SECRET_ACCESS_KEY=synthetic-secret\n",
                encoding="utf-8",
            )
            self.assertEqual(
                MODULE.s3_auth(argparse.Namespace(s3_env_file=env_file)),
                (None, "synthetic-access", "synthetic-secret"),
            )
            with self.assertRaises(ValueError):
                MODULE.s3_auth(
                    argparse.Namespace(
                        s3_env_file=env_file,
                        vault_auth_role="role",
                        s3_access_key_vault_ref="vault:team/data/s3#access_key",
                        s3_secret_key_vault_ref="vault:team/data/s3#secret_key",
                    )
                )

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
                "{type: model_registry, mrid: qwen/revision, dst: /hf_models/}",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                base_preset=base,
                time_limit=None,
                flavor=None,
                bucket="approved-bucket",
                code_bundle_src="/approved/project/exp685/code",
                code_bundle_file="code_abc.tar.gz",
                code_bundle_sha256="a" * 64,
                code_revision="e" * 40,
                pair_src="/approved/project/exp685/pairs/f0/abc",
                expected_pair_acceptance_sha256="b" * 64,
                expected_pair_runtime_contract_sha256="c" * 64,
                vendor_src="/approved/project/exp685/vendor/run1",
                vendor_file="peft-0.20.0.zip",
                vendor_sha256="d" * 64,
                output_dst="/approved/project/exp685/train/f0/control/run1",
                model_input_line_file=model,
                fold=0,
                mode="paired_hard_control",
                technical_smoke=False,
            )
            model_sha = __import__("hashlib").sha256(
                (model.read_text(encoding="utf-8").strip() + "\n").encode()
            ).hexdigest()
            with mock.patch.object(MODULE, "EXPECTED_MODEL_INPUT_LINE_SHA256", model_sha):
                payload = MODULE.build_train(args)
        self.assertNotIn("type: files", payload)
        self.assertEqual(payload.count("type: s3msk"), 4)
        self.assertIn("when: on_job_status=succeeded", payload)
        self.assertIn("location: cluster", payload)
        self.assertIn("verify_training_artifact.py", payload)
        self.assertNotIn("adapter.zip", payload)
        self.assertNotIn("images/frozen", payload)
        self.assertIn("stage_training_input.py", payload)
        self.assertIn("/work/pair_clean", payload)

    def test_training_rejects_unbraced_model_registry_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.yaml"
            base.write_text(BASE, encoding="utf-8")
            model = root / "model.txt"
            model.write_text(
                "type: model_registry, mrid: qwen/revision, dst: /hf_models/",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                base_preset=base,
                time_limit=None,
                flavor=None,
                bucket="approved-bucket",
                code_bundle_src="/approved/project/exp685/code",
                code_bundle_file="code_abc.tar.gz",
                code_bundle_sha256="a" * 64,
                code_revision="e" * 40,
                pair_src="/approved/project/exp685/pairs/f0/abc",
                expected_pair_acceptance_sha256="b" * 64,
                expected_pair_runtime_contract_sha256="c" * 64,
                vendor_src="/approved/project/exp685/vendor/run1",
                vendor_file="peft-0.20.0.zip",
                vendor_sha256="d" * 64,
                output_dst="/approved/project/exp685/train/f0/control/run1",
                model_input_line_file=model,
                fold=0,
                mode="paired_hard_control",
                technical_smoke=False,
            )
            with self.assertRaises(ValueError):
                MODULE.build_train(args)

    def test_legacy_bridge_has_s3_only_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.yaml"
            base.write_text(BASE, encoding="utf-8")
            args = argparse.Namespace(
                base_preset=base,
                time_limit="20m",
                flavor="8cpu-128ram",
                bucket="approved-bucket",
                code_bundle_src="/approved/project/exp685/code",
                code_bundle_file="code_abc.tar.gz",
                code_bundle_sha256="a" * 64,
                code_revision="e" * 40,
                artifact_src=["0=job0/output0", "3=job3/output3"],
                expected_score_sha=["0=" + "b" * 64, "3=" + "c" * 64],
                expected_archive_sha=["0=" + "d" * 64, "3=" + "e" * 64],
                output_dst="/approved/project/exp685/bridge/run1",
            )
            payload = MODULE.build_bridge(args)
        self.assertEqual(payload.count("type: artifact"), 2)
        self.assertEqual(payload.count("type: s3msk"), 2)
        self.assertNotIn("type: files", payload)
        self.assertIn("bridge_legacy_teacher.py", payload)
        self.assertIn("when: on_job_status=succeeded", payload)

    def test_legacy_bridge_supports_vault_references(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.yaml"
            base.write_text(BASE, encoding="utf-8")
            args = argparse.Namespace(
                base_preset=base,
                time_limit="20m",
                flavor="8cpu-128ram",
                bucket="approved-bucket",
                vault_auth_role="team__ds_data_plane-ro",
                s3_access_key_vault_ref="vault:team/data/s3#access_key",
                s3_secret_key_vault_ref="vault:team/data/s3#secret_key",
                code_bundle_src="/approved/project/exp685/code",
                code_bundle_file="code_abc.tar.gz",
                code_bundle_sha256="a" * 64,
                code_revision="e" * 40,
                artifact_src=["0=job0/output0", "3=job3/output3"],
                expected_score_sha=["0=" + "b" * 64, "3=" + "c" * 64],
                expected_archive_sha=["0=" + "d" * 64, "3=" + "e" * 64],
                output_dst="/approved/project/exp685/bridge/run1",
            )
            payload = MODULE.build_bridge(args)
        self.assertIn('auth_role: "team__ds_data_plane-ro"', payload)
        self.assertEqual(payload.count("access_key: vault:"), 0)
        self.assertEqual(payload.count('access_key: "vault:'), 2)
        self.assertEqual(payload.count('secret_key: "vault:'), 2)
        self.assertNotIn("type: files", payload)

    def test_prepare_uses_frozen_remote_bundles(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.yaml"
            base.write_text(BASE, encoding="utf-8")
            args = argparse.Namespace(
                base_preset=base,
                time_limit="20m",
                flavor="8cpu-128ram",
                bucket="approved-bucket",
                code_bundle_src="/approved/project/exp685/code",
                code_bundle_file="code_abc.tar.gz",
                code_bundle_sha256="a" * 64,
                code_revision="e" * 40,
                source_bundle_src="/approved/project/exp680",
                source_bundle_file="source.tar.gz",
                source_bundle_sha256="b" * 64,
                source_runtime_rel="experiments/641/runtime/fold0",
                teacher_bundle_src="/approved/project/exp662",
                teacher_bundle_file="teacher.tar.gz",
                teacher_bundle_sha256="c" * 64,
                teacher_runtime_rel="runtime/fold0_full",
                teacher_scores_src="/approved/project/exp685/bridge/run1/fold0",
                teacher_artifact_rel="teacher_outer_train_scores.zip",
                output_dst="/approved/project/exp685/r0/fold0/run1",
                fold=0,
            )
            payload = MODULE.build_prepare(args)
        self.assertNotIn("type: files", payload)
        self.assertEqual(payload.count("type: s3msk"), 5)
        self.assertIn("build_pair_runtime.py", payload)
        self.assertIn("verify_pair_runtime.py", payload)
        self.assertIn("cp -a", payload)
        self.assertIn("source_runtime", payload)
        self.assertIn("mkdir -p /work/output &&", payload)
        self.assertNotIn("mkdir -p /work/output/runtime", payload)


if __name__ == "__main__":
    unittest.main()
