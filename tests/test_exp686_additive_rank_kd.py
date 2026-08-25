from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/686_qwen35_4b_additive_rank_kd"


def load(name: str):
    sys.path.insert(0, str(EXP))
    for dependency in (
        "build_pair_runtime",
        "verify_pair_runtime",
        "verify_code_bundle",
        "train_pair_fold",
    ):
        sys.modules.pop(dependency, None)
    spec = importlib.util.spec_from_file_location(f"exp686_{name}", EXP / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load("train_pair_fold")
PROMOTION = load("promotion_gate")
BUILD_CODE = load("build_code_bundle")
VERIFY_CODE = load("verify_code_bundle")
PRESET = load("build_remote_compute_preset")
EVALUATE = load("evaluate")
STAGE_INPUT = load("stage_training_input")


def write_code_acceptance(
    root: Path, *, scope: str, revision: str, bundle_sha: str
) -> Path:
    payload = {
        "schema_version": 1,
        "experiment_id": "686",
        "scope": scope,
        "git_revision": revision,
        "bundle_sha256": bundle_sha,
        "manifest_sha256": "c" * 64,
        "files": 1,
        "directories": 1,
        "source_paths": (
            VERIFY_CODE.TRAINING_SOURCE_PATHS
            if scope == "training"
            else VERIFY_CODE.TRAINING_SOURCE_PATHS
            + VERIFY_CODE.EVALUATION_ONLY_SOURCE_PATHS
        ),
        "decision": "ACCEPT_CODE_BUNDLE",
    }
    payload["acceptance_sha256"] = PRESET.canonical_sha256(payload)
    path = root / f"{scope}_acceptance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_promotion_receipt(
    root: Path, *, stage: str
) -> tuple[Path, str, Path, str]:
    if stage == "blind3":
        folds = [3]
        decision = "OPEN_CONFIRMATION_FOLDS124"
        output_source = "approved/project/exp686/eval/f3"
        parent = {
            "parent_promotion_gate_sha256": None,
            "parent_promotion_gate_file_sha256": None,
            "parent_promotion_evaluation_file_sha256": None,
            "parent_promotion_source": None,
        }
    elif stage == "confirmation":
        folds = [1, 2, 4]
        decision = "OPEN_FULL_REPLAY"
        output_source = "approved/project/exp686/eval/confirmation"
        parent = {
            "parent_promotion_gate_sha256": "d" * 64,
            "parent_promotion_gate_file_sha256": "e" * 64,
            "parent_promotion_evaluation_file_sha256": "f" * 64,
            "parent_promotion_source": "/approved/project/exp686/eval/f3",
        }
    else:
        raise ValueError(stage)
    evaluation = {
        "schema_version": 1,
        "experiment_id": "686",
        "stage": stage,
        "folds": folds,
        "passed": True,
        "decision": decision,
        "validation_labels_read_by_training": 0,
        "sealed_rows": 0,
        "public_used": False,
        "gates": {"frozen_gate": True},
        "evaluation_code_bundle_sha256": "a" * 64,
        "evaluation_code_revision": "b" * 40,
        "evaluation_code_acceptance_sha256": "c" * 64,
        "candidate_acceptance_sha256": ["1" * 64 for _ in folds],
        "control_acceptance_sha256": ["2" * 64 for _ in folds],
        "candidate_prediction_sha256": ["3" * 64 for _ in folds],
        "control_prediction_sha256": ["4" * 64 for _ in folds],
        "replay_bundle_sha256": "5" * 64,
        "replay_contract_sha256": "6" * 64,
        "registry_sha256": "7" * 64,
    }
    evaluation["evaluation_sha256"] = PRESET.canonical_sha256(evaluation)
    evaluation_path = root / f"{stage}_evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    evaluation_file_sha = __import__("hashlib").sha256(
        evaluation_path.read_bytes()
    ).hexdigest()
    gate = {
        **evaluation,
        "evaluation_file_sha256": evaluation_file_sha,
        "source_output_metadata_sha256": "8" * 64,
        "evaluator_job_id": f"eval-{stage}",
        "evaluator_job_state": "SUCCEEDED",
        "evaluation_output_source": output_source,
        "gate_authority": "independent_main_integrator",
        "gate_main_revision": "9" * 40,
        **parent,
    }
    gate.pop("validation_labels_read_by_evaluator", None)
    gate["promotion_gate_sha256"] = PRESET.canonical_sha256(gate)
    gate_path = root / f"{stage}_promotion_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    gate_file_sha = __import__("hashlib").sha256(gate_path.read_bytes()).hexdigest()
    return gate_path, gate_file_sha, evaluation_path, evaluation_file_sha


class AdditiveRankKDTests(unittest.TestCase):
    def test_r0_code_acceptance_is_exactly_bound(self):
        payload = {
            "schema_version": 1,
            "experiment_id": "686",
            "scope": "r0_prepare",
            "git_revision": "a" * 40,
            "bundle_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "files": 6,
            "directories": 7,
            "source_paths": ["frozen.py"],
            "decision": "ACCEPT_R0_CODE_BUNDLE",
        }
        payload["acceptance_sha256"] = STAGE_INPUT.canonical_sha256(payload)
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        accepted = STAGE_INPUT.verify_r0_code_acceptance(
            encoded,
            expected_acceptance_sha256=payload["acceptance_sha256"],
        )
        self.assertEqual(accepted["acceptance_sha256"], payload["acceptance_sha256"])
        with self.assertRaisesRegex(ValueError, "frozen consumer contract"):
            STAGE_INPUT.verify_r0_code_acceptance(
                encoded,
                expected_acceptance_sha256="d" * 64,
            )

    def test_training_input_whitelist_allows_only_bound_r0_code_acceptance(self):
        self.assertIn("r0_code_acceptance.json", STAGE_INPUT.REQUIRED_FILES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in STAGE_INPUT.REQUIRED_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            payloads, ignored = STAGE_INPUT.inventory(root)
            self.assertEqual(set(payloads), set(STAGE_INPUT.REQUIRED_FILES))
            self.assertEqual(ignored, [])
            (root / "unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected remote pair input member"):
                STAGE_INPUT.inventory(root)

    def test_promotion_hash_matches_evaluator_for_unicode_fields(self):
        payload = {
            "category": "Легковоспламеняющиеся",
            "metrics": {"БАД": 0.95},
        }
        self.assertEqual(
            PROMOTION.canonical_sha256(payload),
            TRAIN.canonical_sha256(payload),
        )

    def test_bad_route_gate_requires_exact_prediction_identity(self):
        categories = np.array(["БАД", "БАД", "Легковоспламеняющиеся"])
        folds = np.array([3, 3, 3])
        baseline = np.array([0, 1, 0], dtype=np.int8)
        control = np.array([0, 1, 1], dtype=np.int8)
        candidate = np.array([1, 0, 1], dtype=np.int8)
        identity = EVALUATE.bad_route_identity(
            categories=categories,
            folds=folds,
            folds_scope=(3,),
            baseline_prediction=baseline,
            control_prediction=control,
            candidate_prediction=candidate,
        )
        self.assertFalse(identity["byte_identical"])
        self.assertFalse(identity["direct_control_candidate_equal"])
        self.assertFalse(identity["production_baseline_candidate_equal"])
        self.assertNotEqual(
            identity["control_prediction_sha256"],
            identity["candidate_prediction_sha256"],
        )

    def test_candidate_adds_rank_loss_without_reducing_hard_bce(self):
        source = inspect.getsource(TRAIN.combine_pair_losses)
        self.assertIn('if mode == "paired_hard_control":\n        return hard', source)
        self.assertIn("return hard + RANK_LOSS_WEIGHT * rank", source)
        self.assertNotIn("0.5 * hard", source)
        self.assertEqual(TRAIN.RANK_LOSS_WEIGHT, 0.5)
        self.assertEqual(
            TRAIN.combine_pair_losses(3.0, 2.0, mode="paired_hard_control"),
            3.0,
        )
        self.assertEqual(
            TRAIN.combine_pair_losses(3.0, 2.0, mode="rank_candidate"),
            4.0,
        )

    def test_training_bundle_is_physically_label_packet_free(self):
        self.assertNotIn(
            "validation/semantic_family_v3/folds.csv", BUILD_CODE.TRAINING_BUNDLE_PATHS
        )
        self.assertIn(
            "validation/semantic_family_v3/folds.csv", BUILD_CODE.EVALUATION_BUNDLE_PATHS
        )
        self.assertEqual(
            list(BUILD_CODE.TRAINING_BUNDLE_PATHS), VERIFY_CODE.TRAINING_SOURCE_PATHS
        )
        self.assertEqual(
            list(BUILD_CODE.EVALUATION_BUNDLE_PATHS),
            VERIFY_CODE.TRAINING_SOURCE_PATHS + VERIFY_CODE.EVALUATION_ONLY_SOURCE_PATHS,
        )

    def test_trainer_accepts_only_training_scope(self):
        base = {
            "schema_version": 1,
            "experiment_id": "686",
            "scope": "training",
            "git_revision": "a" * 40,
            "bundle_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "files": 1,
            "directories": 1,
            "source_paths": list(VERIFY_CODE.TRAINING_SOURCE_PATHS),
            "decision": "ACCEPT_CODE_BUNDLE",
        }
        for scope, accepted in (("training", True), ("evaluation", False)):
            payload = dict(base, scope=scope)
            payload["acceptance_sha256"] = TRAIN.canonical_sha256(payload)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "acceptance.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                if accepted:
                    self.assertEqual(TRAIN.load_code_acceptance(path)["scope"], "training")
                else:
                    with self.assertRaisesRegex(ValueError, "code-bundle acceptance"):
                        TRAIN.load_code_acceptance(path)

    def test_blind_fold3_eval_preset_is_cpu_remote_only(self):
        base_text = """job:
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
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.yml"
            base.write_text(base_text, encoding="utf-8")
            acceptance = write_code_acceptance(
                Path(directory), scope="evaluation", revision="b" * 40, bundle_sha="a" * 64
            )
            args = argparse.Namespace(
                base_preset=base,
                time_limit="20m",
                flavor="8cpu-128ram",
                bucket="approved-bucket",
                code_bundle_src="/approved/project/exp686/code",
                code_bundle_file="evaluation.tar.gz",
                code_bundle_sha256="a" * 64,
                code_revision="b" * 40,
                code_acceptance_file=acceptance,
                candidate_output_src="/approved/project/exp686/f3/rank",
                control_output_src="/approved/project/exp686/f3/control",
                source_fold3_src="/approved/project/exp685/r0/f3",
                expected_source_fold3_contract="c" * 64,
                eval_technical_smoke=True,
                promotion_receipt_file=None,
                promotion_evaluation_file=None,
                promotion_src=None,
                expected_promotion_file_sha256=None,
                expected_promotion_evaluation_file_sha256=None,
                output_dst="/approved/project/exp686/eval/f3",
            )
            payload = PRESET.build_fast_outer3_eval(args)
        self.assertIn("flavor: 8cpu-128ram", payload)
        self.assertIn("evaluate_outer3_fast.py", payload)
        self.assertIn("/work/source_fold3/source_runtime", payload)
        self.assertIn("/work/code/validation/semantic_family_v3/folds.csv", payload)
        self.assertEqual(payload.count("type: s3msk"), 5)
        self.assertNotIn("type: model_registry", payload)

    def test_training_preset_rejects_evaluation_scope_before_delivery(self):
        base_text = """job:
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.yml"
            base.write_text(base_text, encoding="utf-8")
            model = root / "model.txt"
            model.write_text(
                "{type: model_registry, mrid: qwen/revision, dst: /hf_models/}",
                encoding="utf-8",
            )
            acceptance = write_code_acceptance(
                root, scope="evaluation", revision="b" * 40, bundle_sha="a" * 64
            )
            args = argparse.Namespace(
                base_preset=base,
                time_limit=None,
                flavor=None,
                bucket="approved-bucket",
                code_bundle_src="/approved/project/exp686/code/evaluation",
                code_bundle_file="evaluation.tar.gz",
                code_bundle_sha256="a" * 64,
                code_revision="b" * 40,
                code_acceptance_file=acceptance,
                pair_src="/approved/project/exp685/pair/f3",
                expected_pair_acceptance_sha256="d" * 64,
                expected_pair_runtime_contract_sha256="e" * 64,
                expected_r0_code_acceptance_sha256="1" * 64,
                vendor_src="/approved/project/exp685/vendor",
                vendor_sha256="f" * 64,
                output_dst="/approved/project/exp686/train/f3/control",
                model_input_line_file=model,
                fold=3,
                mode="paired_hard_control",
                technical_smoke=False,
                promotion_receipt_file=None,
                promotion_evaluation_file=None,
                promotion_src=None,
                expected_promotion_file_sha256=None,
                expected_promotion_evaluation_file_sha256=None,
            )
            with self.assertRaisesRegex(ValueError, "consuming stage"):
                PRESET.build_train(args)

    def test_confirmation_fold_requires_self_hashed_outer3_receipt(self):
        args = argparse.Namespace(
            promotion_receipt_file=None,
            promotion_evaluation_file=None,
            promotion_src=None,
            expected_promotion_file_sha256=None,
            expected_promotion_evaluation_file_sha256=None,
        )
        with self.assertRaisesRegex(ValueError, "requires a frozen promotion"):
            PRESET.validate_promotion_receipt(args, fold=1)

    def test_evaluator_binds_artifacts_to_exact_stage_receipt(self):
        receipt = {"promotion_gate_sha256": "a" * 64}
        source = "/approved/project/exp686/eval/blind3/run1"
        file_sha = "b" * 64
        evaluation_file_sha = "d" * 64
        accepted = {
            fold: {
                "promotion_receipt_sha256": receipt["promotion_gate_sha256"],
                "promotion_receipt_file_sha256": file_sha,
                "promotion_evaluation_file_sha256": evaluation_file_sha,
                "promotion_receipt_source": source,
            }
            for fold in (1, 2, 4)
        }
        EVALUATE.verify_promotion_lineage(
            accepted,
            stage="confirmation",
            promotion=receipt,
            promotion_file_sha256=file_sha,
            promotion_evaluation_file_sha256=evaluation_file_sha,
            promotion_source=source,
        )
        accepted[2]["promotion_receipt_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "promotion lineage mismatch"):
            EVALUATE.verify_promotion_lineage(
                accepted,
                stage="confirmation",
                promotion=receipt,
                promotion_file_sha256=file_sha,
                promotion_evaluation_file_sha256=evaluation_file_sha,
                promotion_source=source,
            )

    def test_stage_topology_excludes_development_fold_from_confirmation(self):
        self.assertEqual(PRESET.STAGE_FOLDS["blind3"], (3,))
        self.assertEqual(PRESET.STAGE_FOLDS["confirmation"], (1, 2, 4))
        self.assertEqual(PRESET.STAGE_FOLDS["full"], (0, 1, 2, 3, 4))

    def test_eval_presets_mount_exact_candidate_and_control_sets(self):
        base_text = """job:
  time_limit: 2h0m0s
  flavor: 8cpu-128ram
  region: msk
  image: example/image:immutable
  preemption: never
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: false
    PYTORCH_ALLOC_CONF: expandable_segments:True
"""
        expected_s3_counts = {"blind3": 5, "confirmation": 10, "full": 14}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.yml"
            base.write_text(base_text, encoding="utf-8")
            acceptance = write_code_acceptance(
                root, scope="evaluation", revision="b" * 40, bundle_sha="a" * 64
            )
            for stage, folds in PRESET.STAGE_FOLDS.items():
                receipt = None
                receipt_sha = None
                promotion_evaluation = None
                promotion_evaluation_sha = None
                promotion_src = None
                if stage == "confirmation":
                    (
                        receipt,
                        receipt_sha,
                        promotion_evaluation,
                        promotion_evaluation_sha,
                    ) = write_promotion_receipt(
                        root, stage="blind3"
                    )
                    promotion_src = "/approved/project/exp686/eval/f3"
                elif stage == "full":
                    (
                        receipt,
                        receipt_sha,
                        promotion_evaluation,
                        promotion_evaluation_sha,
                    ) = write_promotion_receipt(
                        root, stage="confirmation"
                    )
                    promotion_src = "/approved/project/exp686/eval/confirmation"
                args = argparse.Namespace(
                    base_preset=base,
                    time_limit="20m",
                    flavor="8cpu-128ram",
                    bucket="approved-bucket",
                    code_bundle_src="/approved/project/exp686/code/evaluation",
                    code_bundle_file="evaluation.tar.gz",
                    code_bundle_sha256="a" * 64,
                    code_revision="b" * 40,
                    code_acceptance_file=acceptance,
                    eval_stage=stage,
                    candidate_src=[
                        f"{fold}=/approved/project/exp686/candidate/f{fold}"
                        for fold in folds
                    ],
                    control_src=[
                        f"{fold}=/approved/project/exp686/control/f{fold}"
                        for fold in folds
                    ],
                    replay_src="/approved/project/exp686/replay",
                    bundle_file="bundle.npz",
                    replay_contract_file="contract.json",
                    registry_file="folds.csv",
                    promotion_receipt_file=receipt,
                    promotion_evaluation_file=promotion_evaluation,
                    promotion_src=promotion_src,
                    expected_promotion_file_sha256=receipt_sha,
                    expected_promotion_evaluation_file_sha256=(
                        promotion_evaluation_sha
                    ),
                    output_dst=f"/approved/project/exp686/eval/{stage}",
                )
                payload = PRESET.build_eval(args)
                self.assertEqual(payload.count("type: s3msk"), expected_s3_counts[stage])
                for fold in folds:
                    self.assertIn(f"/work/candidate/fold{fold}", payload)
                    self.assertIn(f"/work/control/fold{fold}", payload)


if __name__ == "__main__":
    unittest.main()
