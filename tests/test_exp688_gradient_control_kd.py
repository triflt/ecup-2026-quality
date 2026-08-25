from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

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

    def test_packet_has_smoke_verifier_and_no_preset(self):
        self.assertTrue((EXP / "verify_training_artifact.py").is_file())
        self.assertFalse((EXP / "build_remote_compute_preset.py").exists())
        readme = (EXP / "README.md").read_text(encoding="utf-8")
        self.assertIn("No preset, upload or job is authorized", readme)
        self.assertIn("There is no lambda, weight, cap or threshold grid", readme)
        self.assertIn("--technical-smoke", readme)


if __name__ == "__main__":
    unittest.main()
