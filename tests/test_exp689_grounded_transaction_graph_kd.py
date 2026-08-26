from __future__ import annotations

import csv
import json
import sys
from collections.abc import Callable
from copy import deepcopy
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


def make_candidate(**fields: Any) -> dict[str, Any]:
    candidate = {"candidate_id": None, **fields}
    candidate["candidate_id"] = build.sha256_bytes(build.canonical_json_bytes(candidate))
    return candidate


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
    first_image = {
        "reference": f"runtime-image://{record_id}/0",
        "content_sha256": token(f"encoded:{record_id}"),
        "decoded_rgb_sha256": token(f"rgb:{record_id}"),
        "media_type": "image/jpeg",
        "width": 64,
        "height": 64,
    }
    start = text.index("fuel capsule")
    end = start + len("fuel capsule")
    candidates = [
        make_candidate(
            candidate_index=0,
            evidence_kind="text_span",
            source_index=0,
            char_start=start,
            char_end=end,
            image_region=None,
            evidence_sha256=build.sha256_text(text[start:end]),
        )
    ]
    for candidate_index, region in enumerate(("full", "q00", "q01", "q10", "q11"), 1):
        evidence_sha = (
            first_image["decoded_rgb_sha256"]
            if region == "full"
            else token(f"rgb:{record_id}:{region}")
        )
        candidates.append(
            make_candidate(
                candidate_index=candidate_index,
                evidence_kind="image_region",
                source_index=None,
                char_start=None,
                char_end=None,
                image_region=region,
                evidence_sha256=evidence_sha,
            )
        )
    return {
        "schema_version": "exp689_source_row_v1",
        "record_id": record_id,
        "row_token": token(f"row:{record_id}"),
        "component_token": token(component_value),
        "family_token": token(f"family:{record_id}"),
        "stratum": stratum,
        "source_card_sha256": build.sha256_bytes(
            build.canonical_json_bytes({"sources": sources, "first_image": first_image})
        ),
        "sources": sources,
        "first_image": first_image,
        "evidence_candidates": candidates,
    }


def make_runtime_inputs(
    root: Path,
    *,
    raw_exclusions: bool = False,
    overlap_first_component: bool = False,
    missing_first_image: bool = False,
) -> dict[str, Any]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    spec = build.load_spec()
    rows: list[dict[str, Any]] = []
    for stratum in spec["audit"]["strata"]:
        for index in range(100):
            component = "old-670-component-000" if overlap_first_component and not rows else None
            rows.append(make_source_row(stratum, index, component))
    if missing_first_image:
        del rows[0]["first_image"]
    source_rows = inputs / "source_rows.jsonl"
    build.write_jsonl(source_rows, rows)
    source_contract = build.with_self_hash(
        {
            "schema_version": "exp689_source_contract_v2",
            "execution_scope": "remote_remote_compute",
            "source_rows_sha256": build.sha256_file(source_rows),
            "source_row_count": len(rows),
            "runtime_sha256": token("runtime"),
            "data_sha256": token("data"),
            "registry_sha256": token("registry"),
            "builder_revision_sha256": token("builder-revision"),
            "eligibility_universe_sha256": token("eligibility-universe"),
            "stratum_derivation_sha256": token("stratum-derivation"),
            "image_membership_sha256": token("first-image-membership"),
            "image_transform_sha256": token("decoded-rgb-and-quadrants"),
            "candidate_generator_sha256": token("candidate-generator"),
            "opaque_token_scheme": "sha256_utf8_v1",
            "label_fields_present": False,
            "score_fields_present": False,
            "first_image_rows_verified": len(rows),
            "image_fetch_failures": 0,
            "image_decode_failures": 0,
            "image_hash_mismatches": 0,
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
    root: Path, request_path: Path, *, mutate: Callable[[list[dict[str, Any]]], None] | None = None
) -> tuple[Path, Path]:
    requests = read_jsonl(request_path)
    selections = []
    for request in requests:
        text_candidate_id = request["evidence_candidates"][0]["candidate_id"]
        selections.append(
            {
                "schema_version": "exp689_teacher_selection_row_v2",
                "audit_id": request["audit_id"],
                "record_id": request["record_id"],
                "request_row_sha256": request["request_row_sha256"],
                "sold_object": "fuel_consumable",
                "substance": "solid_fuel",
                "relation": "included",
                "object_evidence_candidate_ids": [text_candidate_id],
                "substance_evidence_candidate_ids": [text_candidate_id],
                "relation_evidence_candidate_ids": [text_candidate_id],
                "support_status": "supported",
                "supervise": True,
            }
        )
    if mutate is not None:
        mutate(selections)
    selection_path = root / "inputs" / "teacher_selections.jsonl"
    build.write_jsonl(selection_path, selections)
    manifest = json.loads(
        (root / "outputs" / "prepared" / "selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    contract = build.with_self_hash(
        {
            "schema_version": "exp689_teacher_selection_contract_v2",
            "execution_scope": "remote_remote_compute",
            "teacher_request_sha256": manifest["teacher_request_sha256"],
            "teacher_model_id": "runtime-model-parameter",
            "teacher_model_revision": "runtime-revision-parameter",
            "prompt_sha256": token("teacher-prompt"),
            "decoding_sha256": token("deterministic-decoding"),
            "inference_bundle_sha256": token("inference-bundle"),
            "job_metadata_sha256": token("job-metadata"),
            "selection_rows_sha256": build.sha256_file(selection_path),
            "selection_row_count": len(selections),
            "output_policy": "closed_enums_and_separate_candidate_ids_only_v2",
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


def base_review(index: int, *, image_required: int, substance_errors: int, evidence_failures: int) -> dict[str, Any]:
    outside_image_index = index - image_required
    return {
        "sold_object_correct": True,
        "substance_correct": not (0 <= outside_image_index < substance_errors),
        "relation_correct": True,
        "evidence_supported": not (0 <= outside_image_index < evidence_failures),
        "unsupported_visual_claim": False,
        "contradiction": False,
        "evidence_slice": "image_required" if index < image_required else "text_sufficient",
    }


def make_review_artifacts(
    root: Path,
    frozen_packet: Path,
    *,
    image_required: int = 30,
    substance_errors: int = 0,
    evidence_failures: int = 0,
    relation_disagreements: int = 0,
    incomplete_b: bool = False,
    adjudicate: bool = True,
) -> dict[str, Path | None]:
    frozen_rows = read_jsonl(frozen_packet)
    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    for index, frozen in enumerate(frozen_rows):
        review_a = base_review(
            index,
            image_required=image_required,
            substance_errors=substance_errors,
            evidence_failures=evidence_failures,
        )
        review_b = deepcopy(review_a)
        if index < relation_disagreements:
            review_b["relation_correct"] = False
        rows_a.append(
            {
                "schema_version": "exp689_review_overlay_row_v1",
                "audit_id": frozen["audit_id"],
                "review": review_a,
            }
        )
        rows_b.append(
            {
                "schema_version": "exp689_review_overlay_row_v1",
                "audit_id": frozen["audit_id"],
                "review": review_b,
            }
        )
    if incomplete_b:
        rows_b.pop()
    review_a_path = root / "inputs" / "review_a.jsonl"
    review_b_path = root / "inputs" / "review_b.jsonl"
    build.write_jsonl(review_a_path, rows_a)
    build.write_jsonl(review_b_path, rows_b)

    def reviewer_contract(role: str, overlay_path: Path, count: int) -> dict[str, Any]:
        return build.with_self_hash(
            {
                "schema_version": "exp689_reviewer_contract_v1",
                "target_audit_sha256": build.sha256_file(frozen_packet),
                "rubric_sha256": build.sha256_file(build.RUBRIC_PATH),
                "review_overlay_sha256": build.sha256_file(overlay_path),
                "reviewer_id_sha256": token(f"opaque-reviewer-{role}"),
                "started_at_utc": "2026-08-26T10:00:00Z",
                "completed_at_utc": "2026-08-26T11:00:00Z",
                "review_rows": count,
                "blind_to_other_reviewer": True,
                "blind_to_outcome_labels": True,
                "blind_to_prior_audits": True,
                "self_sha256": None,
            }
        )

    contract_a_path = root / "inputs" / "review_a_contract.json"
    contract_b_path = root / "inputs" / "review_b_contract.json"
    write_json(contract_a_path, reviewer_contract("a", review_a_path, len(rows_a)))
    write_json(contract_b_path, reviewer_contract("b", review_b_path, len(rows_b)))
    result: dict[str, Path | None] = {
        "review_a": review_a_path,
        "review_a_contract": contract_a_path,
        "review_b": review_b_path,
        "review_b_contract": contract_b_path,
        "adjudication": None,
        "adjudication_contract": None,
    }
    if relation_disagreements and adjudicate:
        disagreement_ids = [row["audit_id"] for row in rows_a[:relation_disagreements]]
        adjudication_rows = [
            {
                "schema_version": "exp689_adjudication_overlay_row_v1",
                "audit_id": row["audit_id"],
                "review": deepcopy(row["review"]),
            }
            for row in rows_a[:relation_disagreements]
        ]
        adjudication_path = root / "inputs" / "adjudication.jsonl"
        build.write_jsonl(adjudication_path, adjudication_rows)
        adjudication_contract = build.with_self_hash(
            {
                "schema_version": "exp689_adjudication_contract_v1",
                "target_audit_sha256": build.sha256_file(frozen_packet),
                "rubric_sha256": build.sha256_file(build.RUBRIC_PATH),
                "review_a_overlay_sha256": build.sha256_file(review_a_path),
                "review_b_overlay_sha256": build.sha256_file(review_b_path),
                "adjudication_overlay_sha256": build.sha256_file(adjudication_path),
                "adjudicator_id_sha256": token("opaque-adjudicator"),
                "started_at_utc": "2026-08-26T12:00:00Z",
                "completed_at_utc": "2026-08-26T12:30:00Z",
                "adjudication_rows": len(adjudication_rows),
                "disagreement_audit_ids_sha256": build.sha256_bytes(
                    build.canonical_json_bytes(disagreement_ids)
                ),
                "blind_to_outcome_labels": True,
                "self_sha256": None,
            }
        )
        adjudication_contract_path = root / "inputs" / "adjudication_contract.json"
        write_json(adjudication_contract_path, adjudication_contract)
        result["adjudication"] = adjudication_path
        result["adjudication_contract"] = adjudication_contract_path
    return result


def build_full_packet(tmp_path: Path, mutate: Any = None) -> tuple[Any, ...]:
    paths = make_runtime_inputs(tmp_path)
    manifest_path, request_path = prepare_runtime(paths)
    selection_path, selection_contract_path = make_selections(
        tmp_path, request_path, mutate=mutate
    )
    frozen_packet, packet_contract = materialize_runtime(
        paths, manifest_path, request_path, selection_path, selection_contract_path
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


def validate_runtime(packet: tuple[Any, ...], reviews: dict[str, Path | None]) -> dict[str, Any]:
    (
        paths,
        manifest_path,
        request_path,
        selection_path,
        selection_contract_path,
        frozen_packet,
        packet_contract,
    ) = packet
    return audit.validate(
        remote_root=paths["root"],
        source_rows_path=paths["source_rows"],
        source_contract_path=paths["source_contract"],
        selection_manifest_path=manifest_path,
        teacher_request_path=request_path,
        teacher_selections_path=selection_path,
        teacher_selection_contract_path=selection_contract_path,
        frozen_packet_path=frozen_packet,
        review_a_path=reviews["review_a"],
        review_a_contract_path=reviews["review_a_contract"],
        review_b_path=reviews["review_b"],
        review_b_contract_path=reviews["review_b_contract"],
        adjudication_path=reviews["adjudication"],
        adjudication_contract_path=reviews["adjudication_contract"],
        packet_contract_path=packet_contract,
        exclusion_670_path=paths["exclusion_670"],
        exclusion_672_path=paths["exclusion_672"],
        output_path=paths["root"] / "outputs" / "acceptance.json",
    )


def test_frozen_contract_is_image_aware_dual_blind_and_gpu_closed() -> None:
    spec = build.load_spec()
    assert spec["image_contract"]["evidence_regions"] == ["full", "q00", "q01", "q10", "q11"]
    assert spec["review_contract"]["blind_reviewers_required"] == 2
    assert spec["gates"]["minimum_target_tuple_correct"] == 282
    assert spec["gates"]["minimum_s1_object_relation_correct"] == 95
    assert spec["gates"]["minimum_s2_object_relation_correct"] == 90
    assert spec["gates"]["minimum_s3_object_relation_correct"] == 90
    assert spec["gates"]["minimum_image_required_rows"] == 30
    assert spec["student_gpu"]["authorized"] is False


def test_remote_pipeline_accepts_exact_dual_300_and_never_authorizes_gpu(
    tmp_path: Path,
) -> None:
    packet = build_full_packet(tmp_path)
    reviews = make_review_artifacts(tmp_path, packet[5])
    result = validate_runtime(packet, reviews)
    assert result["status"] == "accepted"
    assert result["metrics"]["target_tuple_correct"] == 300
    assert result["metrics"]["image_required_rows"] == 30
    assert result["metrics"]["relation_cohen_kappa"] == 1.0
    assert result["metrics"]["relation_gwet_ac1"] == 1.0
    assert result["gate_checks"]["relation_gwet_ac1"] is True
    assert result["student_gpu_authorized"] is False


def test_teacher_request_includes_image_and_strips_private_target_fields(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path, raw_exclusions=True)
    manifest_path, request_path = prepare_runtime(paths)
    request = read_jsonl(request_path)[0]
    assert set(request) == {
        "schema_version",
        "audit_id",
        "record_id",
        "source_card_sha256",
        "sources",
        "first_image",
        "evidence_candidates",
        "request_row_sha256",
    }
    assert {candidate["image_region"] for candidate in request["evidence_candidates"] if candidate["evidence_kind"] == "image_region"} == {"full", "q00", "q01", "q10", "q11"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["teacher_visible_family_tokens"] == 0
    assert manifest["teacher_visible_outcome_fields"] == 0


def test_missing_first_image_fails_closed(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path, missing_first_image=True)
    with pytest.raises(build.ContractError, match="first_image"):
        prepare_runtime(paths)


def test_image_failure_counter_fails_closed(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path)
    contract = json.loads(paths["source_contract"].read_text(encoding="utf-8"))
    contract["image_decode_failures"] = 1
    write_json(paths["source_contract"], build.with_self_hash({**contract, "self_sha256": None}))
    with pytest.raises(build.ContractError, match="failures must be zero"):
        prepare_runtime(paths)


def test_unknown_evidence_candidate_id_fails_closed(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path)
    manifest_path, request_path = prepare_runtime(paths)

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["relation_evidence_candidate_ids"] = ["f" * 64]

    selection_path, selection_contract = make_selections(tmp_path, request_path, mutate=mutate)
    with pytest.raises(build.ContractError, match="nonexistent evidence candidate ID"):
        materialize_runtime(
            paths, manifest_path, request_path, selection_path, selection_contract
        )


def test_unsupported_teacher_response_must_clear_all_three_evidence_lists(
    tmp_path: Path,
) -> None:
    paths = make_runtime_inputs(tmp_path)
    manifest_path, request_path = prepare_runtime(paths)

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["support_status"] = "ambiguous"
        rows[0]["supervise"] = False
        rows[0]["object_evidence_candidate_ids"] = []
        rows[0]["substance_evidence_candidate_ids"] = []

    selection_path, selection_contract = make_selections(tmp_path, request_path, mutate=mutate)
    with pytest.raises(build.ContractError, match="must abstain"):
        materialize_runtime(
            paths, manifest_path, request_path, selection_path, selection_contract
        )


def test_incomplete_second_review_fails_closed(tmp_path: Path) -> None:
    packet = build_full_packet(tmp_path)
    reviews = make_review_artifacts(tmp_path, packet[5], incomplete_b=True)
    with pytest.raises(build.ContractError, match="expected exactly 300"):
        validate_runtime(packet, reviews)


def test_unadjudicated_disagreement_fails_closed(tmp_path: Path) -> None:
    packet = build_full_packet(tmp_path)
    reviews = make_review_artifacts(
        tmp_path, packet[5], relation_disagreements=1, adjudicate=False
    )
    with pytest.raises(build.ContractError, match="requires adjudication"):
        validate_runtime(packet, reviews)


def test_low_relation_kappa_rejects_despite_adjudicated_final_pass(tmp_path: Path) -> None:
    packet = build_full_packet(tmp_path)
    reviews = make_review_artifacts(tmp_path, packet[5], relation_disagreements=30)
    result = validate_runtime(packet, reviews)
    assert result["metrics"]["raw_exact_review_agreement"] == 0.9
    assert result["metrics"]["relation_cohen_kappa"] < 0.8
    assert result["metrics"]["relation_gwet_ac1"] > 0.8
    assert result["gate_checks"]["relation_cohen_kappa"] is False
    assert result["gate_checks"]["relation_gwet_ac1"] is True
    assert result["status"] == "rejected_by_frozen_gate"


def test_less_than_30_image_required_rows_rejects(tmp_path: Path) -> None:
    packet = build_full_packet(tmp_path)
    reviews = make_review_artifacts(tmp_path, packet[5], image_required=29)
    result = validate_runtime(packet, reviews)
    assert result["metrics"]["image_required_rows"] == 29
    assert result["gate_checks"]["image_required_rows"] is False
    assert result["status"] == "rejected_by_frozen_gate"


@pytest.mark.parametrize(
    ("substance_errors", "expected_status", "expected_tuple_correct"),
    [(18, "accepted", 282), (19, "rejected_by_frozen_gate", 281)],
)
def test_target_tuple_gate_counts_substance_errors(
    tmp_path: Path,
    substance_errors: int,
    expected_status: str,
    expected_tuple_correct: int,
) -> None:
    packet = build_full_packet(tmp_path)
    reviews = make_review_artifacts(
        tmp_path, packet[5], substance_errors=substance_errors
    )
    result = validate_runtime(packet, reviews)
    assert result["status"] == expected_status
    assert result["metrics"]["target_tuple_correct"] == expected_tuple_correct
    assert result["gate_checks"]["target_tuple_correct"] is (substance_errors == 18)
    assert result["student_gpu_authorized"] is False


def test_four_unsupported_evidence_rows_reject(tmp_path: Path) -> None:
    packet = build_full_packet(tmp_path)
    reviews = make_review_artifacts(tmp_path, packet[5], evidence_failures=4)
    result = validate_runtime(packet, reviews)
    assert result["metrics"]["unsupported_or_incorrect_evidence"] == 4
    assert result["gate_checks"]["evidence_supported"] is False
    assert result["status"] == "rejected_by_frozen_gate"


def test_source_and_teacher_contracts_bind_required_provenance(tmp_path: Path) -> None:
    packet = build_full_packet(tmp_path)
    paths = packet[0]
    source_contract = json.loads(paths["source_contract"].read_text(encoding="utf-8"))
    assert {
        "runtime_sha256",
        "data_sha256",
        "registry_sha256",
        "builder_revision_sha256",
        "eligibility_universe_sha256",
        "stratum_derivation_sha256",
        "image_membership_sha256",
        "image_transform_sha256",
        "candidate_generator_sha256",
    } <= set(source_contract)
    teacher_contract = json.loads(packet[4].read_text(encoding="utf-8"))
    assert {
        "teacher_model_id",
        "teacher_model_revision",
        "prompt_sha256",
        "decoding_sha256",
        "inference_bundle_sha256",
        "job_metadata_sha256",
        "selection_rows_sha256",
    } <= set(teacher_contract)


def test_exclusion_overlap_is_component_only_and_fails_selection(tmp_path: Path) -> None:
    paths = make_runtime_inputs(tmp_path, overlap_first_component=True)
    with pytest.raises(build.ContractError, match="found 99"):
        prepare_runtime(paths)


def test_paths_outside_remote_root_are_rejected(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(build.ContractError, match="below --remote-root"):
        build.require_remote_path(remote_root, outside, context="source rows", must_exist=True)
