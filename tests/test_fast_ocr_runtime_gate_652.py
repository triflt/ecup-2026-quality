from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/652_fast_ocr_runtime_gate/runtime_gate.py"
SPEC = importlib.util.spec_from_file_location("exp652_runtime_gate", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def measurement(**overrides):
    value = {
        "rows": 600,
        "images": 2274,
        "ocr_seconds": 120.0,
        "parseable_fraction": 0.995,
        "bounded_region_fraction": 1.0,
        "critical_span_recall": 0.92,
        "unsupported_text_fraction": 0.005,
    }
    value.update(overrides)
    return value


def test_fast_ocr_passes_only_inside_combined_runtime_and_quality_gate() -> None:
    result = MODULE.evaluate(measurement())
    assert result["decision"] == "PASS"
    assert result["projected_public_minutes"] < 16.0
    assert result["projected_private_minutes"] < 38.0
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("ocr_seconds", 131.0, "ocr_runtime"),
        ("parseable_fraction", 0.98, "parseable"),
        ("bounded_region_fraction", 0.999, "bounded_regions"),
        ("critical_span_recall", 0.899, "critical_span_recall"),
        ("unsupported_text_fraction", 0.011, "unsupported_text"),
    ],
)
def test_fast_ocr_rejects_each_failed_gate(field: str, value: float, check: str) -> None:
    result = MODULE.evaluate(measurement(**{field: value}))
    assert result["decision"] == "REJECT"
    assert result["checks"][check] is False


def test_fast_ocr_rejects_a_different_runtime_sample() -> None:
    with pytest.raises(ValueError, match="differs from frozen control"):
        MODULE.evaluate(measurement(images=2273))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ocr_seconds", -1.0),
        ("ocr_seconds", float("nan")),
        ("parseable_fraction", 1.01),
        ("critical_span_recall", float("inf")),
        ("unsupported_text_fraction", -0.01),
    ],
)
def test_fast_ocr_rejects_invalid_measurements(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        MODULE.evaluate(measurement(**{field: value}))
