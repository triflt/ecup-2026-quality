from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "621_semantic_v3_fisher_blockwise_merge"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


merge = _load("exp621_merge_test", EXP / "merge_utils.py")
preflight = _load("exp621_preflight_test", EXP / "preflight.py")


def test_effective_delta_is_b_times_a_and_metadata_is_strict() -> None:
    a = np.eye(2)
    b = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    delta = merge.effective_lora_delta(a, b, lora_alpha=4.0, rank=2)
    np.testing.assert_allclose(delta, 2.0 * b)
    metadata = merge.AdapterMetadata("base", "rev", ("q",), 2, 4.0)
    merge.validate_compatible_metadata(metadata, metadata)
    with pytest.raises(ValueError, match="base_model_revision"):
        merge.validate_compatible_metadata(
            metadata, metadata.__class__("base", "other", ("q",), 2, 4.0)
        )


def test_fisher_coefficients_are_sorted_conservative_and_fail_closed() -> None:
    values = merge.compute_blockwise_alpha({"z": 3.0, "a": 1.0}, {"z": 1.0, "a": 9.0})
    assert list(values) == ["a", "z"]
    assert values["a"] == pytest.approx(0.5)
    assert values["z"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="invalid Fisher"):
        merge.compute_blockwise_alpha({"a": -1.0}, {"a": 1.0})


def test_merge_uses_effective_deltas_not_factor_averaging() -> None:
    original = {"q": np.eye(2)}
    specialist = {"q": np.full((2, 2), 3.0)}
    merged = merge.merge_effective_deltas(original, specialist, {"q": 0.5})
    np.testing.assert_allclose(merged["q"], np.asarray([[2.0, 1.5], [1.5, 2.0]]))
    with pytest.raises(ValueError, match="identical modules"):
        merge.merge_effective_deltas(original, {"v": specialist["q"]}, {"q": 0.5})


def test_deterministic_svd_and_factorization_have_fixed_signs() -> None:
    matrix = np.asarray([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    first, error = merge.deterministic_svd_compress(matrix, rank=1)
    second, error_again = merge.deterministic_svd_compress(matrix, rank=1)
    np.testing.assert_array_equal(first, second)
    assert error == error_again
    a, b, factor_error = merge.factorize_delta(matrix, rank=2, lora_alpha=2.0)
    np.testing.assert_allclose((2.0 / 2.0) * (b @ a), matrix)
    assert factor_error == pytest.approx(0.0)


def test_label_blind_manifest_never_accepts_supervision_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "category": ["A", "B"],
            "semantic_component": ["c0", "c1"],
            "component_size": [1, 1],
            "split": ["development", "development"],
            "development_fold": [0, 1],
            "label": [0, 1],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="forbidden supervision"):
        preflight.read_label_blind_manifest(path)


def test_label_blind_audit_and_public_metadata_checks(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "category": ["A", "B", "A"],
            "semantic_component": ["c0", "c1", "c2"],
            "component_size": [1, 1, 1],
            "split": ["development", "development", "sealed_holdout"],
            "development_fold": [0, 1, -1],
        }
    ).to_csv(path, index=False)
    report = preflight.load_and_audit_manifest(path)
    assert report["labels_loaded"] is False
    assert report["sealed_rows_loaded"] == 0
    preflight.validate_public_metadata(json.loads((EXP / "results" / "metrics.json").read_text()))
    with pytest.raises(ValueError, match="forbidden private token"):
        preflight.validate_public_metadata({"path": "private://bucket/file"})


def test_public_scaffold_has_no_completed_metrics_claim() -> None:
    metrics = json.loads((EXP / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "screen_running"
    assert metrics["validation_complete"] is False
    assert metrics["folds"] == []
    assert metrics["metrics"] == {}
