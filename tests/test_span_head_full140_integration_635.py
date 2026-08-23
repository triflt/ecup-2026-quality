from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/635_span_head_full140_integration"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _module("exp635_evaluator_test", EXP / "evaluate.py")


def _evidence_bundle() -> dict[str, np.ndarray]:
    return {
        "seed632_evidence": np.asarray(["точная фраза", "NO_EVIDENCE"]),
        "seed632_concept": np.asarray(["COMPOSITION", "NO_EVIDENCE"]),
        "seed632_char_start": np.asarray([4, -1], dtype=np.int64),
        "seed632_char_end": np.asarray([16, -1], dtype=np.int64),
        "canonical_text": np.asarray(["это точная фраза здесь", "нет данных"]),
        "qwen35_original_logit": np.asarray([-2.0, -2.0]),
        "qwen35_seed632_logit": np.asarray([2.0, 2.0]),
    }


def test_spec_pins_full_140_and_exactly_four_variants() -> None:
    spec = evaluator.load_spec()
    evaluator.verify_source_recipe(spec)
    assert spec["parent_full_system"] == "140"
    assert spec["experiment_603_allowed_as_parent"] is False
    assert spec["component_f1_compared_to_full_ensemble"] is False
    assert spec["variants"] == [evaluator.BASELINE, *evaluator.CANDIDATES]
    assert spec["route"]["БАД"]["threshold"] == 0.27193570137023926
    assert spec["route"][evaluator.FLAMMABLE]["threshold"] == 0.953912615776062


def test_accepts_only_terminal_five_fold_632_report(tmp_path: Path) -> None:
    report = {
        "protocol": "632_independent_seed_full_five_fold_v1",
        "candidate_experiment_id": "632",
        "candidate_seed": 31415,
        "folds_evaluated": list(evaluator.FOLDS),
        "winning_folds": 4,
        "passed": True,
        "sealed_rows_loaded": 0,
        "public_used": False,
    }
    path = tmp_path / "accepted-632.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    observed = evaluator.verify_accepted_632(
        path, {"accepted_632_report_sha256": evaluator.sha256_file(path)}
    )
    assert observed == report
    report["winning_folds"] = 3
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="five-fold gate"):
        evaluator.verify_accepted_632(
            path, {"accepted_632_report_sha256": evaluator.sha256_file(path)}
        )


def test_evidence_gate_uses_632_only_for_verified_exact_substring() -> None:
    bundle = _evidence_bundle()
    valid = evaluator.structural_evidence_mask(bundle)
    variants = evaluator.variant_probabilities(bundle, valid)
    assert valid.tolist() == [True, False]
    assert variants["evidence_gated_632"][0] == pytest.approx(variants["replace_with_632"][0])
    assert variants["evidence_gated_632"][1] == pytest.approx(variants[evaluator.BASELINE][1])
    assert variants["fixed_mean_original_632"] == pytest.approx(
        0.5 * variants[evaluator.BASELINE] + 0.5 * variants["replace_with_632"]
    )


def test_non_exact_or_malformed_evidence_fails_closed() -> None:
    bundle = _evidence_bundle()
    bundle["seed632_evidence"][0] = "выдуманная фраза"
    with pytest.raises(ValueError, match="non-exact"):
        evaluator.structural_evidence_mask(bundle)
    bundle = _evidence_bundle()
    bundle["seed632_char_start"][1] = 0
    with pytest.raises(ValueError, match="malformed NO_EVIDENCE"):
        evaluator.structural_evidence_mask(bundle)


def test_same_prior_override_is_applied_after_every_route_variant() -> None:
    rows = len(evaluator.FOLDS) * len(evaluator.CATEGORIES) * 3
    folds = np.repeat(np.asarray(evaluator.FOLDS), len(evaluator.CATEGORIES) * 3)
    categories = np.tile(np.repeat(np.asarray(evaluator.CATEGORIES), 3), len(evaluator.FOLDS))
    bundle = {
        "categories": categories,
        "folds": folds,
        "robust_base_score": np.linspace(-1, 1, rows),
        "qwen3vl_score": np.linspace(1, -1, rows),
        "prior_override": np.full(rows, -1, dtype=np.int8),
    }
    bundle["prior_override"][::4] = 1
    spec = evaluator.load_spec()
    low, _ = evaluator.route_predictions(
        bundle=bundle,
        qwen_probability=np.zeros(rows),
        route=spec["route"],
    )
    high, _ = evaluator.route_predictions(
        bundle=bundle,
        qwen_probability=np.ones(rows),
        route=spec["route"],
    )
    assert np.all(low[::4] == 1)
    assert np.all(high[::4] == 1)


def test_full_gate_requires_four_fold_wins_and_runtime() -> None:
    metrics = {
        "macro_delta": 0.004,
        "winning_folds": 3,
        "categories": {category: {"delta": 0.001} for category in evaluator.CATEGORIES},
        "corrected": 20,
        "regressed": 10,
        "corrected_to_regressed": 2.0,
        "component_bootstrap": {"probability_delta_positive": 0.95},
        "false_negatives": {
            "flammable": {"delta": 0},
            "all_positive": {"delta": 0},
        },
    }
    runtime = {
        "projected_public_minutes": 19.0,
        "projected_private_minutes": 39.0,
        "optimized_predictions_identical": True,
        "adapter_manifest_verified": True,
    }
    gate = evaluator.load_spec()["acceptance"]["full"]
    gates = evaluator._full_gates(metrics, gate, runtime)
    assert gates["folds_won_at_least_4"] is False
    metrics["winning_folds"] = 4
    runtime["projected_private_minutes"] = 41.0
    gates = evaluator._full_gates(metrics, gate, runtime)
    assert gates["private_runtime_within_40_minutes"] is False


def test_contract_explicitly_rejects_603_substitution(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.npz"
    registry = tmp_path / "folds.csv"
    bundle.write_bytes(b"bundle")
    registry.write_bytes(b"registry")
    spec = evaluator.load_spec()
    provenance = {
        component: {
            "folds": {
                str(fold): {
                    "target_fold_excluded": True,
                    "artifact_sha256": "a" * 64,
                }
                for fold in evaluator.FOLDS
            }
        }
        for component in (
            "robust_base",
            "qwen3vl",
            "qwen35_original",
            "qwen35_seed632",
            "prior_override",
        )
    }
    contract = {
        **spec["required_replay_contract"],
        "experiment_id": "635",
        "bundle_sha256": evaluator.sha256_file(bundle),
        "registry_sha256": evaluator.sha256_file(registry),
        "accepted_632_report_sha256": "b" * 64,
        "source_sha256": spec["source_sha256"],
        "route": spec["route"],
        "component_provenance": provenance,
        "array_schema": {},
    }
    contract["source_experiment_603"] = True
    contract["contract_sha256"] = evaluator.canonical_sha256(contract)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="source_experiment_603"):
        evaluator.verify_replay_contract(
            path=path,
            bundle_path=bundle,
            registry_path=registry,
            spec=spec,
        )
