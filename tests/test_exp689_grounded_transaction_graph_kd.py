from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "689_qwen35_4b_grounded_transaction_graph_kd"
)
sys.path.insert(0, str(EXPERIMENT))

import build_target_audit as build
import validate_target_audit as audit


def token(value: str) -> str:
    return build.sha256_text(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_source_row(stratum: str, index: int, component_value: str | None = None) -> dict[str, Any]:
    record_id = f"record-{stratum}-{index:03d}"
    component_value = component_value or f"new-component-{stratum}-{index:03d}"
    text = f"Synthetic product {record_id} includes a declared fuel capsule."
    sources = [
        {
            "source_index": 0,
            "source_kind": "description",
            "text": text,
            "text_sha256": build.sha256_text(text),
        }
    ]
    start = text.index("fuel capsule")
    end = start + len("fuel capsule")
    return {
        "schema_version": "exp689_source_row_v1",
        "record_id": record_id,
        "row_token": token(f"row:{record_id}"),
        "component_token": token(component_value),
        "family_token": token(f"family:{record_id}"),
        "stratum": stratum,
        "source_card_sha256": build.sha256_bytes(build.canonical_json_bytes(sources)),
        "sources": sources,
        "evidence_candidates": [
            {
                "candidate_index": 0,
                "source_index": 0,
                "char_start": start,
                "char_end": end,
                "evidence_sha256": build.sha256_text(text[start:end]),
            }
        ],
    }


def make_runtime_inputs(
    root: Path,
    *,
    raw_exclusions: bool = False,
    overlap_first_component: bool = False,
) -> dict[str, Any]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    spec = build.load_spec()
    rows: list[dict[str, Any]] = []
    for stratum in spec["audit"]["strata"]:
        for index in range(100):
            component = "old-670-component-000" if overlap_first_component and not rows else None
            rows.append(make_source_row(stratum, index, component))
    source_rows = inputs / "source_rows.jsonl"
    build.write_jsonl(source_rows, rows)
    source_contract = build.with_self_hash(
        {
            "schema_version": "exp689_source_contract_v1",
            "execution_scope": "remote_remote_compute",
            "source_rows_sha256": build.sha256_file(source_rows),
            "source_row_count": len(rows),
            "opaque_token_scheme": "sha256_utf8_v1",
            "label_fields_present": False,
            "score_fields_present": False,
            "sealed_rows": 0,
            "public_rows": 0,
            "self_sha256": None,
        }
    )
    source_contract_path = inputs / "source_contract.json"
    write_json(source_contract_path, source_contract)

    values_670 = [f"old-670-component-{index:03d}" for index in range(300)]
    values_672 = [f"old-672-component-{index:03d}" for index in range(40)]
    if raw_exclusions:
        exclusion_670 = inputs / "exclusion_670.csv"
        with exclusion_670.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["semantic_component"])
            writer.writeheader()
            writer.writerows({"semantic_component": value} for value in values_670)
        exclusion_672 = inputs / "exclusion_672.json"
        write_json(
            exclusion_672,
            {"audit": {"components": [{"component": value} for value in values_672]}},
        )
        adapter_args = {
            "exclusion_670_adapter": "csv_component_column",
            "exclusion_672_adapter": "json_component_list",
            "exclusion_670_component_locator": "semantic_component",
            "exclusion_672_component_locator": "audit.components",
            "exclusion_670_component_value_field": None,
            "exclusion_672_component_value_field": "component",
            "exclusion_670_token_mode": "sha256_utf8_v1",
            "exclusion_672_token_mode": "sha256_utf8_v1",
        }
    else:
        exclusion_670 = inputs / "exclusion_670.json"
        exclusion_672 = inputs / "exclusion_672.json"
        for experiment_id, values, path in (
            ("670", values_670, exclusion_670),
            ("672", values_672, exclusion_672),
        ):
            manifest = build.with_self_hash(
                {
                    "schema_version": "exp689_exclusion_manifest_v1",
                    "experiment_id": experiment_id,
                    "row_tokens": [],
                    "component_tokens": sorted(token(value) for value in values),
                    "family_tokens": [],
                    "self_sha256": None,
                }
            )
            write_json(path, manifest)
        adapter_args = {
            "exclusion_670_adapter": "canonical_json",
            "exclusion_672_adapter": "canonical_json",
            "exclusion_670_component_locator": "component_tokens",
            "exclusion_672_component_locator": "component_tokens",
            "exclusion_670_component_value_field": None,
            "exclusion_672_component_value_field": None,
            "exclusion_670_token_mode": "already_sha256",
            "exclusion_672_token_mode": "already_sha256",
        }
    return {
        "root": root,
        "source_rows": source_rows,
        "source_contract": source_contract_path,
        "exclusion_670": exclusion_670,
        "exclusion_672": exclusion_672,
        "exclusion_670_sha256": build.sha256_file(exclusion_670),
        "exclusion_672_sha256": build.sha256_file(exclusion_672),
        **adapter_args,
    }


def prepare_runtime(paths: dict[str, Any], name: str = "prepared") -> tuple[Path, Path]:
    output_dir = paths["root"] / "outputs" / name
    output_dir.parent.mkdir(exist_ok=True)
    build.prepare(
        remote_root=paths["root"],
        source_rows_path=paths["source_rows"],
        source_contract_path=paths["source_contract"],
        exclusion_670_path=paths["exclusion_670"],
        exclusion_672_path=paths["exclusion_672"],
        exclusion_670_sha256=paths["exclusion_670_sha256"],
        exclusion_672_sha256=paths["exclusion_672_sha256"],
        exclusion_670_adapter=paths["exclusion_670_adapter"],
        exclusion_672_adapter=paths["exclusion_672_adapter"],
        exclusion_670_component_locator=paths["exclusion_670_component_locator"],
        exclusion_672_component_locator=paths["exclusion_672_component_locator"],
        exclusion_670_component_value_field=paths["exclusion_670_component_value_field"],
        exclusion_672_component_value_field=paths["exclusion_672_component_value_field"],
        exclusion_670_token_mode=paths["exclusion_670_token_mode"],
        exclusion_672_token_mode=paths["exclusion_672_token_mode"],
        output_dir=output_dir,
    )
    return output_dir / "selection_manifest.json", output_dir / "teacher_request.jsonl"


def make_selections(
    root: Path,
    request_path: Path,
    *,
    mutate: Any = None,
) -> tuple[Path, Path]:
    requests = read_jsonl(request_path)
    selections = [
        {
            "schema_version": "exp689_teacher_selection_row_v1",
            "audit_id": request["audit_id"],
            "record_id": request["record_id"],
            "request_row_sha256": request["request_row_sha256"],
            "sold_object": "fuel_consumable",
            "substance": "solid_fuel",
            "relation": "included",
            "evidence_candidate_indices": [0],
            "support_status": "supported",
            "supervise": True,
        }
        for request in requests
    ]
    if mutate is not None:
        mutate(selections)
    selection_path = root / "inputs" / "teacher_selections.jsonl"
    build.write_jsonl(selection_path, selections)
    manifest_path = root / "outputs" / "prepared" / "selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = build.with_self_hash(
        {
            "schema_version": "exp689_teacher_selection_contract_v1",
            "execution_scope": "remote_remote_compute",
            "teacher_request_sha256": manifest["teacher_request_sha256"],
            "selection_rows_sha256": build.sha256_file(selection_path),
            "selection_row_count": len(selections),
            "output_policy": "closed_enums_and_candidate_indices_only_v1",
            "free_text_fields": 0,
            "forbidden_fields_present": False,
            "self_sha256": None,
        }
    )
    contract_path = root / "inputs" / "teacher_selection_contract.json"
    write_json(contract_path, contract)
    return selection_path, contract_path


def materialize_runtime(
    paths: dict[str, Any],
    manifest_path: Path,
    request_path: Path,
    selection_path: Path,
    selection_contract_path: Path,
) -> tuple[Path, Path]:
    output_dir = paths["root"] / "outputs" / "frozen_audit"
    build.materialize(
        remote_root=paths["root"],
        selection_manifest_path=manifest_path,
        teacher_request_path=request_path,
        teacher_selections_path=selection_path,
        teacher_selection_contract_path=selection_contract_path,
        output_dir=output_dir,
    )
    return output_dir / "target_audit.jsonl", output_dir / "target_audit_contract.json"


def complete_review(
    frozen_packet: Path,
    completed_path: Path,
    *,
    evidence_failures: int = 0,
    contradictions: int = 0,
    substance_errors: int = 0,
) -> None:
    rows = read_jsonl(frozen_packet)
    for index, row in enumerate(rows):
        row["review"] = {
            "sold_object_correct": True,
            "substance_correct": index >= substance_errors,
            "relation_correct": True,
            "evidence_supported": index >= evidence_failures,
            "contradiction": index < contradictions,
        }
    build.write_jsonl(completed_path, rows)


def validate_runtime(
    paths: dict[str, Any],
    manifest_path: Path,
    request_path: Path,
    selection_path: Path,
    selection_contract_path: Path,
    frozen_packet: Path,
    packet_contract: Path,
    completed_review: Path,
) -> dict[str, Any]:
    return audit.validate(
        remote_root=paths["root"],
        source_rows_path=paths["source_rows"],
        source_contract_path=paths["source_contract"],
        selection_manifest_path=manifest_path,
        teacher_request_path=request_path,
        teacher_selections_path=selection_path,
        teacher_selection_contract_path=selection_contract_path,
        frozen_packet_path=frozen_packet,
        completed_review_path=completed_review,
        packet_contract_path=packet_contract,
        exclusion_670_path=paths["exclusion_670"],
        exclusion_672_path=paths["exclusion_672"],
        output_path=paths["root"] / "outputs" / "acceptance.json",
    )


def build_full_packet(tmp_path: Path, mutate: Any = None) -> tuple[Any, ...]:
    paths = make_runtime_inputs(tmp_path)
    manifest_path, request_path = prepare_runtime(paths)
    selection_path, selection_contract_path = make_selections(
        tmp_path, request_path, mutate=mutate
    )
    frozen_packet, packet_contract = materialize_runtime(
        paths,
        manifest_path,
        request_path,
        selection_path,
        selection_contract_path,
    )
    return (
        paths,
        manifest_path,
        request_path,
        selection_path,
        selection_contract_path,
        frozen_packet,
        packet_contract,
    )


def test_frozen_enums_and_gate_thresholds_are_exact() -> None:
    spec = build.load_spec()
    assert spec["enums"]["sold_object"] == [
        "device",
        "fuel_consumable",
        "accessory",
        "kit",
        "other",
        "unknown",
    ]
    assert spec["enums"]["substance"] == [
        "gas",
        "flammable_liquid",
        "solid_fuel",
        "ignition_aid",
        "none",
        "unknown",
    ]
    assert spec["gates"] == {
        "maximum_contradictions": 15,
        "minimum_critical_joint_sold_object_relation": 95,
        "minimum_evidence_supported": 297,
        "minimum_target_tuple_correct": 282,
        "minimum_overall_supervised": 225,
        "minimum_rare_positive_supervised": 80,
        "required_nonduplicate_rows": 300,
        "required_schema_grounding": 300,
    }
    assert spec["student_gpu"]["authorized"] is False


def test_remote_pipeline_accepts_exact_300_and_never_authorizes_gpu(tmp_path: Path) -> None:
    packet = build_full_packet(tmp_path)
    completed = tmp_path / "inputs" / "completed_review.jsonl"
    complete_review(packet[5], completed)
    result = validate_runtime(*packet[:5], packet[5], packet[6], completed)
    assert result["status"] == "accepted"
    assert result["decision"] == "READY_FOR_SEPARATE_STUDENT_GPU_GO"
    assert result["student_gpu_authorized"] is False
    assert result["metrics"]["target_tuple_correct"] == 300
    assert result["metrics"]["sold_object_relation_correct"] == 300
    assert result["metrics"]["critical_joint_sold_object_relation"] == 100
    assert result["gate_checks"]["component_disjoint_from_670_672"] is True


def test_teacher_request_strips_stratum_family_and_outcome_fields(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path, raw_exclusions=True)
    manifest_path, request_path = prepare_runtime(paths)
    requests = read_jsonl(request_path)
    assert len(requests) == 300
    forbidden = set(build.load_spec()["teacher_output_policy"]["forbidden_fields"])
    assert set(requests[0]) == {
        "schema_version",
        "audit_id",
        "record_id",
        "source_card_sha256",
        "sources",
        "evidence_candidates",
        "request_row_sha256",
    }
    serialized_keys = {key for row in requests for key in row}
    assert serialized_keys.isdisjoint(forbidden)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["exclusion_670_binding"]["component_count"] == 300
    assert manifest["exclusion_672_binding"]["component_count"] == 40
    assert manifest["teacher_visible_family_tokens"] == 0


def test_unsupported_and_ambiguous_must_abstain(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path)
    manifest_path, request_path = prepare_runtime(paths)

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["support_status"] = "ambiguous"
        rows[0]["supervise"] = True
        rows[0]["evidence_candidate_indices"] = []

    selection_path, selection_contract = make_selections(tmp_path, request_path, mutate=mutate)
    with pytest.raises(build.ContractError, match="must abstain"):
        materialize_runtime(
            paths,
            manifest_path,
            request_path,
            selection_path,
            selection_contract,
        )


def test_forbidden_teacher_field_fails_closed(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path)
    manifest_path, request_path = prepare_runtime(paths)

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["label"] = 1

    selection_path, selection_contract = make_selections(tmp_path, request_path, mutate=mutate)
    with pytest.raises(build.ContractError, match="exact fields required"):
        materialize_runtime(
            paths,
            manifest_path,
            request_path,
            selection_path,
            selection_contract,
        )


def test_exclusion_overlap_leaves_stratum_short_and_fails(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path, overlap_first_component=True)
    with pytest.raises(build.ContractError, match="found 99"):
        prepare_runtime(paths)


def test_wrong_remote_exclusion_sha_fails_before_selection(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path)
    paths["exclusion_670_sha256"] = "0" * 64
    with pytest.raises(build.ContractError, match="source SHA mismatch"):
        prepare_runtime(paths)


def test_completed_review_cannot_change_teacher_target(tmp_path: Path) -> None:
    packet = build_full_packet(tmp_path)
    completed = tmp_path / "inputs" / "completed_review.jsonl"
    complete_review(packet[5], completed)
    rows = read_jsonl(completed)
    rows[0]["target"]["relation"] = "absent"
    build.write_jsonl(completed, rows)
    with pytest.raises(build.ContractError, match="immutable field target changed"):
        validate_runtime(*packet[:5], packet[5], packet[6], completed)


def test_evidence_gate_rejects_four_unsupported_claims(tmp_path: Path) -> None:
    packet = build_full_packet(tmp_path)
    completed = tmp_path / "inputs" / "completed_review.jsonl"
    complete_review(packet[5], completed, evidence_failures=4)
    result = validate_runtime(*packet[:5], packet[5], packet[6], completed)
    assert result["status"] == "rejected_by_frozen_gate"
    assert result["decision"] == "NO_GO"
    assert result["metrics"]["evidence_supported"] == 296
    assert result["gate_checks"]["evidence_supported"] is False
    assert result["student_gpu_authorized"] is False


@pytest.mark.parametrize(
    ("substance_errors", "expected_status", "expected_tuple_correct"),
    [
        (18, "accepted", 282),
        (19, "rejected_by_frozen_gate", 281),
    ],
)
def test_target_tuple_gate_counts_substance_errors(
    tmp_path: Path,
    substance_errors: int,
    expected_status: str,
    expected_tuple_correct: int,
) -> None:
    packet = build_full_packet(tmp_path)
    completed = tmp_path / "inputs" / "completed_review.jsonl"
    complete_review(packet[5], completed, substance_errors=substance_errors)
    result = validate_runtime(*packet[:5], packet[5], packet[6], completed)
    assert result["status"] == expected_status
    assert result["metrics"]["target_tuple_correct"] == expected_tuple_correct
    assert result["metrics"]["sold_object_relation_correct"] == 300
    assert result["gate_checks"]["target_tuple_correct"] is (substance_errors == 18)
    assert result["gate_checks"]["critical_joint_sold_object_relation"] is True
    assert result["student_gpu_authorized"] is False


def test_supported_response_cannot_select_nonexistent_candidate(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path)
    manifest_path, request_path = prepare_runtime(paths)

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["evidence_candidate_indices"] = [1]

    selection_path, selection_contract = make_selections(tmp_path, request_path, mutate=mutate)
    with pytest.raises(build.ContractError, match="nonexistent evidence"):
        materialize_runtime(
            paths,
            manifest_path,
            request_path,
            selection_path,
            selection_contract,
        )


def test_paths_outside_remote_root_are_rejected(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(build.ContractError, match="below --remote-root"):
        build.require_remote_path(
            remote_root,
            outside,
            context="source rows",
            must_exist=True,
        )
