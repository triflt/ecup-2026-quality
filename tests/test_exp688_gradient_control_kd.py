from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/688_qwen35_4b_gradient_control_kd"


def load_train_module():
    for path in (
        EXP,
        ROOT / "experiments/687_qwen35_4b_gradient_conflict_probe",
        ROOT / "experiments/686_qwen35_4b_additive_rank_kd",
        ROOT / "experiments/645_qwen_scale_2x3_gate",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    for dependency in (
        "build_pair_runtime",
        "gradient_metrics",
        "train_lora",
        "train_pair_fold",
        "verify_probe_artifact",
    ):
        sys.modules.pop(dependency, None)
    spec = importlib.util.spec_from_file_location(
        "exp688_train_gradient_control", EXP / "train_gradient_control.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load_train_module()


def load_exp688_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CODE_VERIFY = load_exp688_module("exp688_verify_code_bundle", "verify_code_bundle.py")
PAIR_VERIFY = load_exp688_module("exp688_verify_paired_smoke", "verify_paired_smoke.py")
PRESET = load_exp688_module("exp688_build_remote_compute_preset", "build_remote_compute_preset.py")


class GradientControlKDTests(unittest.TestCase):
    def test_exactly_two_terminal_selectable_candidate_modes(self):
        self.assertEqual(
            TRAIN.CANDIDATE_MODES,
            (
                "asymmetric_hard_primary_pcgrad",
                "hard_anchored_norm_cap",
            ),
        )
        self.assertEqual(
            TRAIN.CANDIDATE_BY_PROBE_DECISION,
            {
                "OPEN_ASYMMETRIC_PCGRAD_SCREEN": (
                    "asymmetric_hard_primary_pcgrad"
                ),
                "ROUTE_MAGNITUDE_CONTROL": "hard_anchored_norm_cap",
            },
        )
        self.assertEqual(
            set(TRAIN.TRAINING_MODES),
            {"paired_hard_control", *TRAIN.CANDIDATE_MODES},
        )

    def test_asymmetric_pcgrad_projects_only_rank_on_conflict(self):
        hard = [np.array([1.0, 0.0], dtype=np.float64)]
        rank = [np.array([-1.0, 1.0], dtype=np.float64)]
        combined, diagnostic = TRAIN.combine_gradients(
            hard,
            rank,
            mode=TRAIN.PCGRAD_MODE,
            selected_candidate_mode=TRAIN.PCGRAD_MODE,
        )
        np.testing.assert_allclose(combined[0], np.array([1.0, 0.5]))
        self.assertTrue(diagnostic["conflict"])
        self.assertTrue(diagnostic["projection_applied"])
        self.assertAlmostEqual(
            diagnostic["projection_retention"], 1.0 / np.sqrt(2.0)
        )
        self.assertAlmostEqual(diagnostic["hard_rank_cosine"], -1.0 / np.sqrt(2.0))
        diagnostic["optimizer_step"] = 1
        TRAIN.verify_diagnostic_row(
            diagnostic,
            optimizer_step=1,
            mode=TRAIN.PCGRAD_MODE,
            selected_candidate_mode=TRAIN.PCGRAD_MODE,
        )

    def test_asymmetric_pcgrad_keeps_nonconflicting_rank_unchanged(self):
        hard = [np.array([1.0, 0.0])]
        rank = [np.array([1.0, 2.0])]
        combined, diagnostic = TRAIN.combine_gradients(
            hard,
            rank,
            mode=TRAIN.PCGRAD_MODE,
            selected_candidate_mode=TRAIN.PCGRAD_MODE,
        )
        np.testing.assert_allclose(combined[0], np.array([1.5, 1.0]))
        self.assertFalse(diagnostic["conflict"])
        self.assertFalse(diagnostic["projection_applied"])
        self.assertEqual(diagnostic["projection_retention"], 1.0)

    def test_hard_anchored_norm_cap_formula(self):
        hard = [np.array([3.0, 4.0])]
        rank = [np.array([6.0, 8.0])]
        combined, diagnostic = TRAIN.combine_gradients(
            hard,
            rank,
            mode=TRAIN.NORM_CAP_MODE,
            selected_candidate_mode=TRAIN.NORM_CAP_MODE,
        )
        # ||h||=5, ||0.5r||=5, cap=2.5, hence scale=0.5.
        np.testing.assert_allclose(combined[0], np.array([4.5, 6.0]))
        self.assertAlmostEqual(diagnostic["norm_cap_limit"], 2.5)
        self.assertAlmostEqual(diagnostic["norm_cap_scale"], 0.5)
        self.assertTrue(diagnostic["norm_cap_applied"])
        self.assertAlmostEqual(diagnostic["selected_rank_grad_norm"], 2.5)
        diagnostic["optimizer_step"] = 1
        TRAIN.verify_diagnostic_row(
            diagnostic,
            optimizer_step=1,
            mode=TRAIN.NORM_CAP_MODE,
            selected_candidate_mode=TRAIN.NORM_CAP_MODE,
        )

    def test_hard_control_is_exact_hard_gradient_for_arbitrary_rank(self):
        hard = [
            np.array([1.25, -2.0], dtype=np.float64),
            np.array([[3.0, -4.5]], dtype=np.float64),
        ]
        rank = [
            np.array([-9.0, 7.0], dtype=np.float64),
            np.array([[11.0, 13.0]], dtype=np.float64),
        ]
        originals = [value.copy() for value in hard]
        combined, diagnostic = TRAIN.combine_gradients(
            hard,
            rank,
            mode=TRAIN.CONTROL_MODE,
            selected_candidate_mode=TRAIN.PCGRAD_MODE,
        )
        for actual, expected in zip(combined, originals, strict=True):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(diagnostic["selected_rank_retention"], 0.0)
        self.assertEqual(diagnostic["selected_rank_grad_norm"], 0.0)
        self.assertAlmostEqual(
            diagnostic["combined_grad_norm_preclip"],
            np.sqrt(sum(float(np.sum(value * value)) for value in originals)),
        )

    def test_randomized_gradient_safety_invariants(self):
        rng = np.random.default_rng(20260825)
        for _ in range(128):
            hard = rng.normal(size=97)
            rank = rng.normal(size=97)

            pcgrad, _ = TRAIN.combine_gradients(
                [hard],
                [rank],
                mode=TRAIN.PCGRAD_MODE,
                selected_candidate_mode=TRAIN.PCGRAD_MODE,
            )
            retained_rank = (pcgrad[0] - hard) / TRAIN.RANK_WEIGHT
            if float(np.dot(hard, rank)) < 0.0:
                self.assertGreaterEqual(float(np.dot(hard, retained_rank)), -1e-8)

            capped, _ = TRAIN.combine_gradients(
                [hard],
                [rank],
                mode=TRAIN.NORM_CAP_MODE,
                selected_candidate_mode=TRAIN.NORM_CAP_MODE,
            )
            applied_rank = capped[0] - hard
            self.assertLessEqual(
                float(np.linalg.norm(applied_rank)),
                TRAIN.NORM_CAP_RATIO * float(np.linalg.norm(hard)) + 1e-8,
            )

    def test_control_and_candidate_share_the_same_autograd_accumulator(self):
        signature = inspect.signature(TRAIN.accumulate_effective_batch_gradients)
        self.assertNotIn("mode", signature.parameters)
        source = inspect.getsource(TRAIN.accumulate_effective_batch_gradients)
        self.assertEqual(source.count(".backward("), 2)
        self.assertIn("(hard_loss / GRADIENT_ACCUMULATION_PAIRS).backward", source)
        self.assertIn("(rank_loss / GRADIENT_ACCUMULATION_PAIRS).backward", source)
        self.assertEqual(TRAIN.GRADIENT_ACCUMULATION_PAIRS, 8)
        self.assertEqual(TRAIN.EFFECTIVE_BATCH_ROWS, 16)

    def test_mode_selection_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "terminal-selected"):
            TRAIN.validate_training_mode(TRAIN.NORM_CAP_MODE, TRAIN.PCGRAD_MODE)
        with self.assertRaisesRegex(ValueError, "did not select"):
            TRAIN.validate_training_mode(TRAIN.CONTROL_MODE, "unknown")

    def test_nonfinite_zero_and_shape_mismatch_fail_closed(self):
        with self.assertRaises(FloatingPointError):
            TRAIN.combine_gradients(
                [np.array([1.0, np.nan])],
                [np.array([1.0, 2.0])],
                mode=TRAIN.PCGRAD_MODE,
                selected_candidate_mode=TRAIN.PCGRAD_MODE,
            )
        with self.assertRaises(FloatingPointError):
            TRAIN.combine_gradients(
                [np.zeros(2)],
                [np.ones(2)],
                mode=TRAIN.PCGRAD_MODE,
                selected_candidate_mode=TRAIN.PCGRAD_MODE,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            TRAIN.combine_gradients(
                [np.ones(2)],
                [np.ones(3)],
                mode=TRAIN.PCGRAD_MODE,
                selected_candidate_mode=TRAIN.PCGRAD_MODE,
            )

    def test_diagnostic_verifier_rejects_formula_tampering(self):
        _, diagnostic = TRAIN.combine_gradients(
            [np.array([1.0, 0.0])],
            [np.array([-1.0, 1.0])],
            mode=TRAIN.PCGRAD_MODE,
            selected_candidate_mode=TRAIN.PCGRAD_MODE,
        )
        diagnostic["optimizer_step"] = 1
        diagnostic["projection_retention"] = 0.99
        with self.assertRaisesRegex(ValueError, "projection_retention"):
            TRAIN.verify_diagnostic_row(
                diagnostic,
                optimizer_step=1,
                mode=TRAIN.PCGRAD_MODE,
                selected_candidate_mode=TRAIN.PCGRAD_MODE,
            )

    def test_diagnostic_verifier_rejects_clip_return_drift(self):
        _, diagnostic = TRAIN.combine_gradients(
            [np.array([1.0, 0.0])],
            [np.array([-1.0, 1.0])],
            mode=TRAIN.PCGRAD_MODE,
            selected_candidate_mode=TRAIN.PCGRAD_MODE,
        )
        diagnostic["optimizer_step"] = 1
        diagnostic["clip_grad_norm_return"] *= 1.1
        with self.assertRaisesRegex(ValueError, "clip_grad_norm_return"):
            TRAIN.verify_diagnostic_row(
                diagnostic,
                optimizer_step=1,
                mode=TRAIN.PCGRAD_MODE,
                selected_candidate_mode=TRAIN.PCGRAD_MODE,
            )

    def test_current_selector_lineage_is_fold3_only_and_fail_closed(self):
        args = SimpleNamespace(fold=2)
        with self.assertRaisesRegex(ValueError, "fold3 only"):
            TRAIN._load_frozen_inputs(args)

    def test_current_inputs_reject_mismatched_selector_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            code_bundle = Path(directory) / "code.tar.gz"
            code_bundle.write_bytes(b"exp688")
            code_sha = TRAIN.sha256_file(code_bundle)
            args = SimpleNamespace(
                fold=3,
                runtime_backend="legacy_eager",
                micro_batch_size_override=2,
                model_revision=TRAIN.MODEL_REVISION,
                code_revision="a" * 40,
                expected_code_bundle_sha256=code_sha,
                code_bundle=code_bundle,
                probe_report=Path("report"),
                probe_acceptance=Path("acceptance"),
                mode=TRAIN.CONTROL_MODE,
                runtime_dir=Path("runtime"),
                pair_runtime=Path("pair"),
                transport_acceptance=Path("transport"),
                parent_code_acceptance=Path("parent-code"),
                probe_code_acceptance=Path("probe-code"),
                vendor_acceptance=Path("vendor-acceptance"),
                vendor_archive=Path("vendor.zip"),
            )
            selection = {
                "selected_candidate_mode": TRAIN.PCGRAD_MODE,
                "pair_runtime_contract_sha256": "1" * 64,
                "pair_runtime_acceptance_sha256": "2" * 64,
                "source_runtime_contract_sha256": "3" * 64,
                "parent_code_bundle_sha256": "4" * 64,
                "parent_code_revision": "5" * 40,
                "parent_code_acceptance_sha256": "6" * 64,
                "probe_code_bundle_sha256": "a" * 64,
                "probe_code_revision": "b" * 40,
                "probe_code_acceptance_sha256": "c" * 64,
                "vendor_zip_sha256": "7" * 64,
                "vendor_bridge_sha256": "8" * 64,
            }
            pair_acceptance = {
                "runtime_contract_sha256": "9" * 64,
                "acceptance_sha256": "2" * 64,
            }
            source_audit = {"contract_sha256": "3" * 64}
            parent_code = {
                "bundle_sha256": "4" * 64,
                "git_revision": "5" * 40,
                "acceptance_sha256": "6" * 64,
            }
            probe_code = {
                "bundle_sha256": "a" * 64,
                "git_revision": "b" * 40,
                "acceptance_sha256": "c" * 64,
            }
            vendor = {
                "vendor_zip_sha256": "7" * 64,
                "bridge_sha256": "8" * 64,
            }
            with (
                mock.patch.object(
                    TRAIN, "load_terminal_probe_selection", return_value=selection
                ),
                mock.patch.object(
                    TRAIN.parent,
                    "load_inputs",
                    return_value=(
                        [None] * TRAIN.EXPECTED_TRAIN_ROWS,
                        [],
                        [None] * TRAIN.EXPECTED_PAIRS,
                        pair_acceptance,
                        source_audit,
                    ),
                ),
                mock.patch.object(
                    TRAIN.parent, "load_transport_acceptance", return_value={}
                ),
                mock.patch.object(
                    TRAIN.parent, "load_code_acceptance", return_value=parent_code
                ),
                mock.patch.object(
                    TRAIN, "load_probe_code_acceptance", return_value=probe_code
                ),
                mock.patch.object(
                    TRAIN.parent, "load_vendor_acceptance", return_value=vendor
                ),
                self.assertRaisesRegex(ValueError, "lineage mismatch"),
            ):
                TRAIN._load_frozen_inputs(args)

    def test_current_inputs_reject_mismatched_probe_code_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            code_bundle = Path(directory) / "code.tar.gz"
            code_bundle.write_bytes(b"exp688")
            args = SimpleNamespace(
                fold=3,
                runtime_backend="legacy_eager",
                micro_batch_size_override=2,
                model_revision=TRAIN.MODEL_REVISION,
                code_revision="a" * 40,
                expected_code_bundle_sha256=TRAIN.sha256_file(code_bundle),
                code_bundle=code_bundle,
                probe_report=Path("report"),
                probe_acceptance=Path("acceptance"),
                mode=TRAIN.CONTROL_MODE,
                runtime_dir=Path("runtime"),
                pair_runtime=Path("pair"),
                transport_acceptance=Path("transport"),
                parent_code_acceptance=Path("parent-code"),
                probe_code_acceptance=Path("probe-code"),
                vendor_acceptance=Path("vendor-acceptance"),
                vendor_archive=Path("vendor.zip"),
            )
            selection = {
                "selected_candidate_mode": TRAIN.PCGRAD_MODE,
                "pair_runtime_contract_sha256": "1" * 64,
                "pair_runtime_acceptance_sha256": "2" * 64,
                "source_runtime_contract_sha256": "3" * 64,
                "parent_code_bundle_sha256": "4" * 64,
                "parent_code_revision": "5" * 40,
                "parent_code_acceptance_sha256": "6" * 64,
                "probe_code_bundle_sha256": "7" * 64,
                "probe_code_revision": "8" * 40,
                "probe_code_acceptance_sha256": "9" * 64,
                "vendor_zip_sha256": "a" * 64,
                "vendor_bridge_sha256": "b" * 64,
            }
            pair_acceptance = {
                "runtime_contract_sha256": "1" * 64,
                "acceptance_sha256": "2" * 64,
            }
            source_audit = {"contract_sha256": "3" * 64}
            parent_code = {
                "bundle_sha256": "4" * 64,
                "git_revision": "5" * 40,
                "acceptance_sha256": "6" * 64,
            }
            probe_code = {
                "bundle_sha256": "c" * 64,
                "git_revision": "8" * 40,
                "acceptance_sha256": "9" * 64,
            }
            vendor = {
                "vendor_zip_sha256": "a" * 64,
                "bridge_sha256": "b" * 64,
            }
            with (
                mock.patch.object(
                    TRAIN, "load_terminal_probe_selection", return_value=selection
                ),
                mock.patch.object(
                    TRAIN.parent,
                    "load_inputs",
                    return_value=(
                        [None] * TRAIN.EXPECTED_TRAIN_ROWS,
                        [],
                        [None] * TRAIN.EXPECTED_PAIRS,
                        pair_acceptance,
                        source_audit,
                    ),
                ),
                mock.patch.object(
                    TRAIN.parent, "load_transport_acceptance", return_value={}
                ),
                mock.patch.object(
                    TRAIN.parent, "load_code_acceptance", return_value=parent_code
                ),
                mock.patch.object(
                    TRAIN, "load_probe_code_acceptance", return_value=probe_code
                ),
                mock.patch.object(
                    TRAIN.parent, "load_vendor_acceptance", return_value=vendor
                ),
                self.assertRaisesRegex(ValueError, "lineage mismatch"),
            ):
                TRAIN._load_frozen_inputs(args)

    def test_code_bundle_verifier_rejects_unknown_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            archive = Path(directory) / "bundle.tar.gz"
            archive.write_bytes(b"immutable archive identity")
            files = {}
            for relative in CODE_VERIFY.SOURCE_PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
                files[relative] = {
                    "size": path.stat().st_size,
                    "sha256": CODE_VERIFY.sha256_file(path),
                }
            directories = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_dir()
            )
            manifest = {
                "schema_version": 1,
                "experiment_id": "688",
                "parent_experiment_id": "686",
                "selector_experiment_id": "687",
                "scope": CODE_VERIFY.SCOPE,
                "git_revision": "a" * 40,
                "source_paths": CODE_VERIFY.SOURCE_PATHS,
                "files": files,
                "directories": directories,
            }
            manifest["manifest_sha256"] = CODE_VERIFY.canonical_sha256(manifest)
            (root / CODE_VERIFY.MANIFEST_NAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            accepted = CODE_VERIFY.verify(
                root,
                archive,
                expected_revision="a" * 40,
                expected_bundle_sha256=CODE_VERIFY.sha256_file(archive),
            )
            self.assertEqual(accepted["decision"], "ACCEPT_EXP688_CODE_BUNDLE")
            (root / "unknown.py").write_text("pass", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown file"):
                CODE_VERIFY.verify(
                    root,
                    archive,
                    expected_revision="a" * 40,
                    expected_bundle_sha256=CODE_VERIFY.sha256_file(archive),
                )

    def test_paired_verifier_rejects_cross_arm_provenance_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "control").mkdir()
            (root / "candidate").mkdir()
            (root / "code_acceptance.json").write_text("{}", encoding="utf-8")
            base = {field: "same" for field in PAIR_VERIFY.PARITY_FIELDS}
            base.update(
                {
                    "mode": TRAIN.CONTROL_MODE,
                    "applied_rank_weight": 0.0,
                    "code_revision": "a" * 40,
                    "code_bundle_sha256": "b" * 64,
                }
            )
            candidate = dict(base)
            candidate.update(
                {
                    "mode": TRAIN.PCGRAD_MODE,
                    "applied_rank_weight": 0.5,
                    "initial_trainable_state_sha256": "drift",
                }
            )
            diagnostic = {field: "same" for field in PAIR_VERIFY.SHARED_DIAGNOSTIC_FIELDS}
            arms = {
                TRAIN.CONTROL_MODE: {"contract": base, "diagnostic": diagnostic},
                TRAIN.PCGRAD_MODE: {
                    "contract": candidate,
                    "diagnostic": diagnostic,
                },
            }
            with (
                mock.patch.object(
                    PAIR_VERIFY,
                    "load_terminal_probe_selection",
                    return_value={"selected_candidate_mode": TRAIN.PCGRAD_MODE},
                ),
                mock.patch.object(
                    PAIR_VERIFY,
                    "_load_arm",
                    side_effect=lambda _path, mode, _selected: arms[mode],
                ),
                self.assertRaisesRegex(ValueError, "parity mismatch"),
            ):
                PAIR_VERIFY.verify(
                    root,
                    probe_report=Path("report"),
                    probe_acceptance=Path("acceptance"),
                    code_acceptance=root / "code_acceptance.json",
                )

    def test_paired_verifier_rejects_raw_gradient_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "control").mkdir()
            (root / "candidate").mkdir()
            (root / "code_acceptance.json").write_text("{}", encoding="utf-8")
            base = {field: "same" for field in PAIR_VERIFY.PARITY_FIELDS}
            base.update(
                {
                    "mode": TRAIN.CONTROL_MODE,
                    "applied_rank_weight": 0.0,
                    "code_revision": "a" * 40,
                    "code_bundle_sha256": "b" * 64,
                }
            )
            candidate = dict(base)
            candidate.update(
                {"mode": TRAIN.PCGRAD_MODE, "applied_rank_weight": 0.5}
            )
            control_diagnostic = {
                field: "same" for field in PAIR_VERIFY.SHARED_DIAGNOSTIC_FIELDS
            }
            candidate_diagnostic = dict(control_diagnostic)
            candidate_diagnostic["hard_rank_dot"] = "drift"
            arms = {
                TRAIN.CONTROL_MODE: {
                    "contract": base,
                    "diagnostic": control_diagnostic,
                },
                TRAIN.PCGRAD_MODE: {
                    "contract": candidate,
                    "diagnostic": candidate_diagnostic,
                },
            }
            with (
                mock.patch.object(
                    PAIR_VERIFY,
                    "load_terminal_probe_selection",
                    return_value={"selected_candidate_mode": TRAIN.PCGRAD_MODE},
                ),
                mock.patch.object(
                    PAIR_VERIFY,
                    "_load_arm",
                    side_effect=lambda _path, mode, _selected: arms[mode],
                ),
                self.assertRaisesRegex(ValueError, "gradient geometry differs"),
            ):
                PAIR_VERIFY.verify(
                    root,
                    probe_report=Path("report"),
                    probe_acceptance=Path("acceptance"),
                    code_acceptance=root / "code_acceptance.json",
                )

    def test_preset_command_is_one_sequential_paired_h100_smoke(self):
        args = SimpleNamespace(
            parent_bundle_file="parent.tar.gz",
            probe_bundle_file="probe.tar.gz",
            exp688_bundle_file="exp688.tar.gz",
            parent_bundle_sha256="1" * 64,
            probe_bundle_sha256="2" * 64,
            exp688_bundle_sha256="3" * 64,
            parent_code_revision="4" * 40,
            probe_code_revision="5" * 40,
            exp688_code_revision="6" * 40,
            probe_report_file_sha256="7" * 64,
            probe_acceptance_file_sha256="8" * 64,
            vendor_sha256="9" * 64,
            expected_pair_acceptance_sha256="a" * 64,
            expected_pair_runtime_contract_sha256="b" * 64,
            expected_r0_code_acceptance_sha256="c" * 64,
        )
        command = PRESET._command(args, TRAIN.PCGRAD_MODE)
        self.assertEqual(command.count("train_gradient_control.py"), 2)
        self.assertEqual(command.count("--technical-smoke"), 4)
        self.assertIn("--mode paired_hard_control", command)
        self.assertIn(f"--mode {TRAIN.PCGRAD_MODE}", command)
        self.assertIn("--output-dir /work/output/control", command)
        self.assertIn("--output-dir /work/output/candidate", command)
        self.assertIn("verify_paired_smoke.py", command)
        self.assertNotIn("compute job submit", command)

    def test_tracked_preset_builder_cannot_read_or_serialize_credentials(self):
        source = inspect.getsource(PRESET)
        for forbidden in (
            "access_key",
            "secret_key",
            "SECRET_ACCESS_KEY",
            "s3_env_file",
            "raw_output",
            "overrides_output",
        ):
            self.assertNotIn(forbidden, source)
        destinations = {action.dest for action in PRESET.parser()._actions}
        self.assertFalse(
            destinations
            & {
                "s3_env_file",
                "raw_output",
                "overrides_output",
                "access_key",
                "secret_key",
            }
        )
        generated = "\n".join(
            [
                *PRESET._input_spec(
                    name="input",
                    bucket="bucket",
                    src="/approved/user/ecup/input",
                    dst="/work/input",
                ),
                *PRESET._output_spec(
                    name="output",
                    bucket="bucket",
                    dst="/approved/user/ecup/output",
                ),
            ]
        )
        self.assertNotIn("access_key", generated)
        self.assertNotIn("secret_key", generated)

    def test_preset_builder_writes_nothing_before_terminal_selector_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                repo=ROOT,
                clean_output=root / "clean.yml",
                expected_region="test-region",
                approved_prefix="/approved/user/ecup",
                parent_bundle_sha256="1" * 64,
                probe_bundle_sha256="2" * 64,
                exp688_bundle_sha256="3" * 64,
                expected_pair_acceptance_sha256="4" * 64,
                expected_pair_runtime_contract_sha256="5" * 64,
                expected_r0_code_acceptance_sha256="6" * 64,
                vendor_sha256="7" * 64,
                probe_report_file_sha256="8" * 64,
                probe_acceptance_file_sha256="8" * 64,
                parent_code_revision="9" * 40,
                probe_code_revision="a" * 40,
                exp688_code_revision="b" * 40,
                parent_bundle_file="parent.tar.gz",
                probe_bundle_file="probe.tar.gz",
                exp688_bundle_file="exp688.tar.gz",
                parent_bundle_src="/approved/user/ecup/experiments/686/code/x",
                probe_bundle_src="/approved/user/ecup/experiments/687/code/x",
                exp688_bundle_src="/approved/user/ecup/experiments/688/code/x",
                pair_src="/approved/user/ecup/experiments/686/pair/x",
                vendor_src="/approved/user/ecup/vendor/x",
                probe_artifact_src=(
                    "/approved/user/ecup/experiments/687/probe/fold3/x"
                ),
                output_dst=(
                    "/approved/user/ecup/experiments/688/technical_smoke/fold3/x"
                ),
                parent_code_acceptance=root / "parent.json",
                probe_code_acceptance=root / "probe-code.json",
                exp688_code_acceptance=root / "exp688-code.json",
                probe_report=root / "gradient_conflict_report.json",
                probe_acceptance=root / "acceptance.json",
            )
            with (
                mock.patch.object(PRESET, "_validate_code_acceptance"),
                mock.patch.object(PRESET, "sha256_file", return_value="8" * 64),
                mock.patch.object(
                    PRESET,
                    "load_terminal_probe_selection",
                    side_effect=ValueError("selector rejected"),
                ),
                self.assertRaisesRegex(ValueError, "selector rejected"),
            ):
                PRESET.build(args)
            self.assertFalse(any(root.glob("*.yml")))

    def test_preset_builder_rejects_selector_probe_code_drift_before_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                repo=ROOT,
                clean_output=root / "clean.yml",
                expected_region="test-region",
                approved_prefix="/approved/user/ecup",
                parent_bundle_sha256="1" * 64,
                probe_bundle_sha256="2" * 64,
                exp688_bundle_sha256="3" * 64,
                expected_pair_acceptance_sha256="4" * 64,
                expected_pair_runtime_contract_sha256="5" * 64,
                expected_r0_code_acceptance_sha256="6" * 64,
                vendor_sha256="7" * 64,
                probe_report_file_sha256="8" * 64,
                probe_acceptance_file_sha256="8" * 64,
                parent_code_revision="9" * 40,
                probe_code_revision="a" * 40,
                exp688_code_revision="b" * 40,
                parent_bundle_file="parent.tar.gz",
                probe_bundle_file="probe.tar.gz",
                exp688_bundle_file="exp688.tar.gz",
                parent_bundle_src="/approved/user/ecup/experiments/686/code/x",
                probe_bundle_src="/approved/user/ecup/experiments/687/code/x",
                exp688_bundle_src="/approved/user/ecup/experiments/688/code/x",
                pair_src="/approved/user/ecup/experiments/686/pair/x",
                vendor_src="/approved/user/ecup/vendor/x",
                probe_artifact_src=(
                    "/approved/user/ecup/experiments/687/probe/fold3/x"
                ),
                output_dst=(
                    "/approved/user/ecup/experiments/688/technical_smoke/fold3/x"
                ),
                parent_code_acceptance=root / "parent.json",
                probe_code_acceptance=root / "probe-code.json",
                exp688_code_acceptance=root / "exp688-code.json",
                probe_report=root / "gradient_conflict_report.json",
                probe_acceptance=root / "acceptance.json",
            )
            parent_acceptance = {"acceptance_sha256": "c" * 64}
            probe_acceptance = {"acceptance_sha256": "d" * 64}
            selection = {
                "selected_candidate_mode": TRAIN.PCGRAD_MODE,
                "pair_runtime_contract_sha256": "5" * 64,
                "pair_runtime_acceptance_sha256": "4" * 64,
                "parent_code_bundle_sha256": "1" * 64,
                "parent_code_revision": "9" * 40,
                "parent_code_acceptance_sha256": "c" * 64,
                "probe_code_bundle_sha256": "e" * 64,
                "probe_code_revision": "a" * 40,
                "probe_code_acceptance_sha256": "d" * 64,
                "vendor_zip_sha256": "7" * 64,
            }
            with (
                mock.patch.object(
                    PRESET,
                    "_validate_code_acceptance",
                    side_effect=[parent_acceptance, probe_acceptance, {}],
                ),
                mock.patch.object(PRESET, "sha256_file", return_value="8" * 64),
                mock.patch.object(
                    PRESET, "load_terminal_probe_selection", return_value=selection
                ),
                self.assertRaisesRegex(ValueError, "selector/preset input mismatch"),
            ):
                PRESET.build(args)
            self.assertFalse(any(root.glob("*.yml")))

    def test_frozen_parent_science_contract(self):
        self.assertEqual(TRAIN.PARENT_EXPERIMENT_ID, "686")
        self.assertEqual(TRAIN.MODEL_ID, "Qwen/Qwen3.5-4B")
        self.assertEqual(
            TRAIN.MODEL_REVISION,
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        )
        self.assertEqual(TRAIN.SEED, 42)
        self.assertEqual(TRAIN.LEARNING_RATE, 2e-4)
        self.assertEqual(TRAIN.EXPECTED_PAIRS, 5440)
        self.assertEqual(TRAIN.EXPECTED_UPDATES, 680)
        self.assertEqual(TRAIN.RANK_WEIGHT, 0.5)
        self.assertEqual(TRAIN.NORM_CAP_RATIO, 0.5)
        self.assertEqual(TRAIN.CANDIDATE_MODES, tuple(TRAIN.CANDIDATE_MODES))

    def test_packet_has_gated_builder_but_no_generated_preset(self):
        self.assertTrue((EXP / "verify_training_artifact.py").is_file())
        self.assertTrue((EXP / "verify_paired_smoke.py").is_file())
        self.assertTrue((EXP / "build_code_bundle.py").is_file())
        self.assertTrue((EXP / "verify_code_bundle.py").is_file())
        self.assertTrue((EXP / "build_remote_compute_preset.py").is_file())
        self.assertFalse(any(EXP.glob("*.yml")))
        self.assertFalse(any(EXP.glob("*.yaml")))
        readme = (EXP / "README.md").read_text(encoding="utf-8")
        self.assertIn("No generated preset, upload or job is authorized", readme)
        self.assertIn("There is no lambda, weight, cap or threshold grid", readme)
        self.assertIn("--technical-smoke", readme)


if __name__ == "__main__":
    unittest.main()
