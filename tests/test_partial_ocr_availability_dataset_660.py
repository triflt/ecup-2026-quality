from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "660_partial_ocr_availability_dataset"
    / "build_partial_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("partial_ocr_660", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeBuilder:
    @staticmethod
    def repair_reasons(source: dict) -> list[str]:
        return list(source.get("reasons", []))

    @staticmethod
    def canonicalize_success_record(source: dict, manifest_row: dict):
        if source.get("invalid"):
            raise ValueError("invalid geometry")
        return {
            "dataset_version": "old",
            "id": manifest_row["id"],
            "regions": source.get("regions", []),
        }, 0


def test_available_record_is_reversioned() -> None:
    status, reasons, record = MODULE.classify_source(
        FakeBuilder(), {"error": None, "regions": [{"text": "x"}]}, {"id": "a"}
    )
    assert status == "OCR_AVAILABLE"
    assert reasons == []
    assert record is not None
    assert record["dataset_version"] == MODULE.DATASET_VERSION


def test_truncated_record_is_unavailable_without_payload() -> None:
    status, reasons, record = MODULE.classify_source(
        FakeBuilder(),
        {"error": None, "reasons": ["generation_reached_limit_without_eos"]},
        {"id": "a"},
    )
    assert status == "OCR_UNAVAILABLE"
    assert reasons == ["generation_reached_limit_without_eos"]
    assert record is None


def test_schema_failure_is_unavailable() -> None:
    status, reasons, record = MODULE.classify_source(
        FakeBuilder(), {"error": None, "invalid": True}, {"id": "a"}
    )
    assert status == "OCR_UNAVAILABLE"
    assert reasons == ["strict_schema_or_geometry_failure"]
    assert record is None

