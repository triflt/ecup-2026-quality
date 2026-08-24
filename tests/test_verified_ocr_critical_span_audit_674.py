from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/674_verified_ocr_critical_span_audit"


def _module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stable_key_is_deterministic() -> None:
    builder = _module("exp674_builder_test", "build_audit.py")
    assert builder.stable_key("x", 1) == builder.stable_key("x", 1)
    assert builder.stable_key("x", 1) != builder.stable_key("x", 2)


def test_evaluator_waits_on_blank_review(tmp_path: Path) -> None:
    evaluator = _module("exp674_evaluator_test", "evaluate_review.py")
    packet = tmp_path / "packet.csv"
    reviews = tmp_path / "reviews.csv"
    mapping = {}
    with packet.open("w", encoding="utf-8", newline="") as pstream, reviews.open(
        "w", encoding="utf-8", newline=""
    ) as rstream:
        pw = csv.DictWriter(pstream, fieldnames=["audit_id"])
        rw = csv.DictWriter(rstream, fieldnames=["audit_id", *evaluator.FIELDS])
        pw.writeheader()
        rw.writeheader()
        for index in range(120):
            audit_id = f"E674-{index:03d}"
            pw.writerow({"audit_id": audit_id})
            rw.writerow({"audit_id": audit_id})
            mapping[audit_id] = {}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "exp674_private_manifest_v1",
                "blind_packet_sha256": evaluator.sha256_file(packet),
                "mapping": mapping,
            }
        ),
        encoding="utf-8",
    )
    result = evaluator.evaluate(
        packet=packet,
        reviews=reviews,
        private_manifest=manifest,
        output=tmp_path / "score.json",
    )
    assert result["decision"] == "WAIT_FOR_COMPLETE_120_ROW_REVIEW"
