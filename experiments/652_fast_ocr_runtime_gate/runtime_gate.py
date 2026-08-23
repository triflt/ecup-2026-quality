from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SPEC_PATH = Path(__file__).with_name("frozen_spec.json")


def load_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def evaluate(measurement: dict[str, Any]) -> dict[str, Any]:
    spec = load_spec()
    control = spec["control"]
    gate = spec["gate"]
    expected = {
        "rows": control["runtime_rows"],
        "images": control["runtime_images"],
    }
    mismatch = {
        key: {"expected": value, "actual": measurement.get(key)}
        for key, value in expected.items()
        if measurement.get(key) != value
    }
    if mismatch:
        raise ValueError(f"runtime sample differs from frozen control: {mismatch}")

    ocr_seconds = float(measurement["ocr_seconds"])
    if not math.isfinite(ocr_seconds) or ocr_seconds < 0:
        raise ValueError("ocr_seconds must be finite and non-negative")

    fractions = {
        key: float(measurement[key])
        for key in (
            "parseable_fraction",
            "bounded_region_fraction",
            "critical_span_recall",
            "unsupported_text_fraction",
        )
    }
    invalid_fractions = {
        key: value
        for key, value in fractions.items()
        if not math.isfinite(value) or not 0.0 <= value <= 1.0
    }
    if invalid_fractions:
        raise ValueError(f"fractions must be finite and within [0, 1]: {invalid_fractions}")

    combined_seconds = float(control["runtime_seconds"]) + ocr_seconds
    public_minutes = combined_seconds * gate["public_rows"] / expected["rows"] / 60
    private_minutes = combined_seconds * gate["private_rows"] / expected["rows"] / 60
    checks = {
        "ocr_runtime": ocr_seconds <= gate["maximum_ocr_seconds"],
        "combined_public_runtime": public_minutes <= gate["maximum_combined_public_minutes"],
        "combined_private_runtime": private_minutes <= gate["maximum_combined_private_minutes"],
        "parseable": fractions["parseable_fraction"] >= gate["minimum_parseable_fraction"],
        "bounded_regions": fractions["bounded_region_fraction"]
        >= gate["minimum_bounded_region_fraction"],
        "critical_span_recall": fractions["critical_span_recall"]
        >= gate["minimum_critical_span_recall"],
        "unsupported_text": fractions["unsupported_text_fraction"]
        <= gate["maximum_unsupported_text_fraction"],
    }
    return {
        "experiment_id": spec["experiment_id"],
        "ocr_seconds": ocr_seconds,
        "combined_seconds_600_rows": combined_seconds,
        "projected_public_minutes": public_minutes,
        "projected_private_minutes": private_minutes,
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "REJECT",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen fast-OCR runtime gate.")
    parser.add_argument("--measurement", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.measurement.read_text(encoding="utf-8")))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
