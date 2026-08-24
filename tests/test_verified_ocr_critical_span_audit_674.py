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


def test_merge_reviews_preserves_template_order(tmp_path: Path) -> None:
    merger = _module("exp674_merge_test", "merge_reviews.py")
    template = tmp_path / "template.csv"
    chunks = [tmp_path / "a.csv", tmp_path / "b.csv"]
    rows = []
    for audit_id in ("E674-001", "E674-002", "E674-003"):
        rows.append(
            {
                "audit_id": audit_id,
                "review_visual_critical_span_present": "no",
                "review_ocr_captures_all_critical_text": "na",
                "review_ocr_preserves_scope_relation": "na",
                "review_unsupported_critical_text": "no",
                "review_evidence_relevant": "no",
                "review_notes": "",
            }
        )
    with template.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=merger.FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row["audit_id"] if field == "audit_id" else "" for field in row})
    for path, subset in zip(chunks, ([rows[1]], [rows[2], rows[0]]), strict=True):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=merger.FIELDS)
            writer.writeheader()
            writer.writerows(subset)
    output = tmp_path / "merged.csv"
    assert merger.merge(template=template, chunks=chunks, output=output) == 3
    assert [row["audit_id"] for row in csv.DictReader(output.open(encoding="utf-8"))] == [
        "E674-001",
        "E674-002",
        "E674-003",
    ]


def test_partial_merge_keeps_unreviewed_template_rows_blank(tmp_path: Path) -> None:
    merger = _module("exp674_partial_merge_test", "merge_reviews.py")
    template = tmp_path / "template.csv"
    chunk = tmp_path / "chunk.csv"
    fields = merger.FIELDS
    blank = {field: "" for field in fields}
    with template.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**blank, "audit_id": "E674-001"})
        writer.writerow({**blank, "audit_id": "E674-002"})
    with chunk.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "E674-002",
                "review_visual_critical_span_present": "no",
                "review_ocr_captures_all_critical_text": "na",
                "review_ocr_preserves_scope_relation": "na",
                "review_unsupported_critical_text": "no",
                "review_evidence_relevant": "no",
                "review_notes": "",
            }
        )
    output = tmp_path / "partial.csv"
    assert merger.merge(template=template, chunks=[chunk], output=output, allow_partial=True) == 1
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["review_visual_critical_span_present"] == ""
    assert rows[1]["review_visual_critical_span_present"] == "no"


def test_evaluator_reports_category_error_and_fold_slices(tmp_path: Path) -> None:
    evaluator = _module("exp674_evaluator_slices_test", "evaluate_review.py")
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
            rw.writerow(
                {
                    "audit_id": audit_id,
                    "review_visual_critical_span_present": "yes",
                    "review_ocr_captures_all_critical_text": "yes",
                    "review_ocr_preserves_scope_relation": "yes",
                    "review_unsupported_critical_text": "no",
                    "review_evidence_relevant": "yes",
                }
            )
            mapping[audit_id] = {
                "category": "БАД" if index < 60 else "Легковоспламеняющиеся",
                "baseline_error": index % 2 == 0,
                "development_fold": index % 5,
            }
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
    assert result["decision"] == "GO_BUILD_OCR_CONSUMER_SCREEN_675_ONLY"
    assert result["metrics"]["by_category"]["БАД"]["rows"] == 60
    assert result["metrics"]["by_baseline_state"]["error"]["rows"] == 60
    assert result["metrics"]["by_fold"]["0"]["rows"] == 24


def test_evaluator_stops_when_remaining_perfect_rows_cannot_restore_recall(
    tmp_path: Path,
) -> None:
    evaluator = _module("exp674_evaluator_early_stop_test", "evaluate_review.py")
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
            row = {"audit_id": audit_id}
            if index < 80:
                visual = index < 69
                row.update(
                    {
                        "review_visual_critical_span_present": "yes" if visual else "no",
                        "review_ocr_captures_all_critical_text": (
                            "yes" if index < 52 else "no" if visual else "na"
                        ),
                        "review_ocr_preserves_scope_relation": (
                            "yes" if index < 66 else "no" if visual else "na"
                        ),
                        "review_unsupported_critical_text": "no",
                        "review_evidence_relevant": "yes" if visual else "no",
                    }
                )
            rw.writerow(row)
            mapping[audit_id] = {
                "category": "БАД" if index < 60 else "Легковоспламеняющиеся",
                "baseline_error": index % 2 == 0,
                "development_fold": index % 5,
            }
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
    assert result["decision"] == "NO_GO_REJECT_VERIFIED_OCR_FEATURE_SCREEN"
    assert result["completed_rows"] == 80
    assert result["human_review_stopped_early"] is True
    assert result["maximum_attainable_if_all_remaining_are_perfect"][
        "critical_span_recall"
    ] == 92 / 109
