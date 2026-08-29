from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/670_causal_attribute_dataset_audit"


def _module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ontology_distinguishes_sold_fuel_from_equipment_and_external_fuel() -> None:
    ontology = _module("exp670_ontology_test", "ontology.py")
    fuel = ontology.extract_causal_target(
        category="Легковоспламеняющиеся",
        name="Газовый баллон пропан 450 г",
        description="Топливо для туристических горелок.",
    )
    assert (fuel.sold_object, fuel.regulated_substance, fuel.relation) == (
        "consumable",
        "gas",
        "sold_object",
    )
    stove = ontology.extract_causal_target(
        category="Легковоспламеняющиеся",
        name="Газовая туристическая горелка",
        description="Работает от цангового баллона, баллон приобретается отдельно.",
    )
    assert stove.sold_object == "device"
    assert stove.regulated_substance == "gas"
    assert stove.relation == "compatible_external"
    assert stove.supported is True


def test_ontology_handles_included_and_negated_relations_with_exact_offsets() -> None:
    ontology = _module("exp670_ontology_test2", "ontology.py")
    included = ontology.extract_causal_target(
        category="Легковоспламеняющиеся",
        name="Набор для пикника",
        description="В комплект входит сухое горючее и металлическая горелка.",
    )
    assert included.sold_object == "kit"
    assert included.regulated_substance == "ignition_aid"
    assert included.relation == "included"
    source = included.evidence_span
    assert source == included.evidence_span
    negated = ontology.extract_causal_target(
        category="Легковоспламеняющиеся",
        name="Плита походная",
        description="Поставляется без газа. Совместима с резьбовым баллоном.",
    )
    assert negated.sold_object == "device"
    assert negated.regulated_substance == "gas"
    assert negated.relation == "negated"


def test_score_blocks_empty_reviews_and_enforces_frozen_gates(tmp_path: Path) -> None:
    evaluator = _module("exp670_evaluate_test", "evaluate_audit.py")
    audit = tmp_path / "audit.csv"
    manifest = tmp_path / "manifest.json"
    fields = [
        "audit_id",
        "focus",
        "row_id",
        "semantic_component",
        "review_sold_object_correct",
        "review_regulated_substance_correct",
        "review_relation_correct",
        "review_evidence_correct",
        "review_unsupported_claim",
    ]
    records = []
    with audit.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(300):
            audit_id = f"C670-{index + 1:03d}"
            row_id = str(index)
            component = f"component-{index}"
            records.append(
                {"audit_id": audit_id, "row_id": row_id, "semantic_component": component}
            )
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "focus": "transaction_scope_priority" if index < 100 else "control",
                    "row_id": row_id,
                    "semantic_component": component,
                    "review_sold_object_correct": "",
                    "review_regulated_substance_correct": "",
                    "review_relation_correct": "",
                    "review_evidence_correct": "",
                    "review_unsupported_claim": "",
                }
            )
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    blocked = evaluator.evaluate(
        audit=audit, private_manifest=manifest, output=tmp_path / "blocked.json"
    )
    assert blocked["decision"] == "WAIT_FOR_COMPLETE_300_ROW_REVIEW"

    rows = list(csv.DictReader(audit.open(encoding="utf-8")))
    with audit.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row.update(
                {
                    "review_sold_object_correct": "yes",
                    "review_regulated_substance_correct": "yes",
                    "review_relation_correct": "yes",
                    "review_evidence_correct": "yes",
                    "review_unsupported_claim": "no",
                }
            )
            writer.writerow(row)
    passed = evaluator.evaluate(
        audit=audit, private_manifest=manifest, output=tmp_path / "passed.json"
    )
    assert passed["decision"] == "GO_BUILD_671"
    assert all(passed["gates"].values())


def test_score_rejects_early_when_unsupported_limit_is_irreversible(tmp_path: Path) -> None:
    evaluator = _module("exp670_evaluate_early_test", "evaluate_audit.py")
    audit = tmp_path / "audit.csv"
    manifest = tmp_path / "manifest.json"
    reviews = tmp_path / "reviews.csv"
    fields = [
        "audit_id",
        "focus",
        "row_id",
        "semantic_component",
        "review_sold_object_correct",
        "review_regulated_substance_correct",
        "review_relation_correct",
        "review_evidence_correct",
        "review_unsupported_claim",
    ]
    records = []
    with audit.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(300):
            audit_id = f"C670-{index + 1:03d}"
            records.append(
                {"audit_id": audit_id, "row_id": str(index), "semantic_component": f"c-{index}"}
            )
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "focus": "transaction_scope_priority" if index < 100 else "control",
                    "row_id": str(index),
                    "semantic_component": f"c-{index}",
                    "review_sold_object_correct": "",
                    "review_regulated_substance_correct": "",
                    "review_relation_correct": "",
                    "review_evidence_correct": "",
                    "review_unsupported_claim": "",
                }
            )
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    review_fields = [
        "audit_id",
        "review_sold_object_correct",
        "review_regulated_substance_correct",
        "review_relation_correct",
        "review_evidence_correct",
        "review_unsupported_claim",
        "review_notes",
    ]
    with reviews.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields)
        writer.writeheader()
        for index in range(4):
            writer.writerow(
                {
                    "audit_id": f"C670-{index + 1:03d}",
                    "review_sold_object_correct": "no",
                    "review_regulated_substance_correct": "yes",
                    "review_relation_correct": "yes",
                    "review_evidence_correct": "yes",
                    "review_unsupported_claim": "yes",
                    "review_notes": "blind review",
                }
            )
    rejected = evaluator.evaluate(
        audit=audit,
        private_manifest=manifest,
        reviews=reviews,
        output=tmp_path / "rejected.json",
    )
    assert rejected["decision"] == "NO_GO_REJECT_CAUSAL_TARGETS"
    assert rejected["irreversible_gates"]["unsupported_claim_limit_exceeded"] is True
