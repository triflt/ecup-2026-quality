"""Validate the dual-blind experiment-689 target audit without authorizing GPU."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from build_target_audit import (
    RUBRIC_PATH,
    SPEC_PATH,
    ContractError,
    canonical_json_bytes,
    expect_exact_keys,
    load_json,
    load_prepared,
    load_runtime_exclusion,
    load_spec,
    load_teacher_selections,
    make_target_row,
    read_jsonl,
    require_hex64,
    require_remote_path,
    sha256_bytes,
    sha256_file,
    validate_self_hash,
    verify_prepared_against_sources,
    with_self_hash,
    write_json,
)

REVIEW_FIELDS = {
    "sold_object_correct",
    "substance_correct",
    "relation_correct",
    "evidence_supported",
    "unsupported_visual_claim",
    "contradiction",
    "evidence_slice",
}
BOOLEAN_REVIEW_FIELDS = REVIEW_FIELDS - {"evidence_slice"}
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def validate_packet_contract(
    contract: dict[str, Any],
    *,
    frozen_packet_path: Path,
    manifest: dict[str, Any],
    selection_contract: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    expect_exact_keys(
        contract,
        {
            "schema_version",
            "experiment_id",
            "execution_scope",
            "frozen_spec_sha256",
            "target_audit_schema_sha256",
            "selection_manifest_self_sha256",
            "teacher_request_sha256",
            "teacher_selection_contract_self_sha256",
            "teacher_selections_sha256",
            "target_audit_sha256",
            "target_audit_rows",
            "selected_counts_by_stratum",
            "gates",
            "embedded_review_fields",
            "student_gpu_authorized",
            "self_sha256",
        },
        "target-audit contract",
    )
    validate_self_hash(contract, "target-audit contract")
    if contract["schema_version"] != "exp689_target_audit_contract_v2":
        raise ContractError("target-audit contract: schema version mismatch")
    if contract["experiment_id"] != "689" or contract["execution_scope"] != "remote_remote_compute":
        raise ContractError("target-audit contract: experiment or execution scope mismatch")
    if contract["frozen_spec_sha256"] != sha256_file(SPEC_PATH):
        raise ContractError("target-audit contract: frozen spec SHA mismatch")
    if contract["target_audit_schema_sha256"] != spec["target_audit_schema_sha256"]:
        raise ContractError("target-audit contract: frozen schema SHA mismatch")
    if contract["selection_manifest_self_sha256"] != manifest["self_sha256"]:
        raise ContractError("target-audit contract: selection lineage mismatch")
    if contract["teacher_request_sha256"] != manifest["teacher_request_sha256"]:
        raise ContractError("target-audit contract: teacher-request lineage mismatch")
    if contract["teacher_selection_contract_self_sha256"] != selection_contract["self_sha256"]:
        raise ContractError("target-audit contract: teacher-contract lineage mismatch")
    if contract["teacher_selections_sha256"] != selection_contract["selection_rows_sha256"]:
        raise ContractError("target-audit contract: teacher-output lineage mismatch")
    if contract["target_audit_sha256"] != sha256_file(frozen_packet_path):
        raise ContractError("target-audit contract: frozen packet SHA mismatch")
    if contract["target_audit_rows"] != spec["audit"]["total_rows"]:
        raise ContractError("target-audit contract: row count mismatch")
    if contract["selected_counts_by_stratum"] != manifest["selected_counts_by_stratum"]:
        raise ContractError("target-audit contract: stratum counts mismatch")
    if contract["gates"] != spec["gates"]:
        raise ContractError("target-audit contract: gates differ from frozen spec")
    if contract["embedded_review_fields"] != 0:
        raise ContractError("target-audit contract: reviews must be separate immutable overlays")
    if contract["student_gpu_authorized"] is not False:
        raise ContractError("target-audit contract: must not authorize student GPU")


def validate_component_disjointness(
    *,
    manifest: dict[str, Any],
    exclusion_670_path: Path,
    exclusion_672_path: Path,
    spec: dict[str, Any],
) -> None:
    runtime_components: dict[str, set[str]] = {}
    for experiment_id, path in (("670", exclusion_670_path), ("672", exclusion_672_path)):
        binding = manifest[f"exclusion_{experiment_id}_binding"]
        sets, actual_binding = load_runtime_exclusion(
            path=path,
            expected_sha256=binding["source_sha256"],
            expected_experiment_id=experiment_id,
            adapter=binding["adapter"],
            component_locator=binding["component_locator"],
            component_value_field=binding["component_value_field"],
            token_mode=binding["token_mode"],
        )
        if actual_binding != binding:
            raise ContractError(
                f"selection manifest: exclusion-{experiment_id} adapter binding mismatch"
            )
        expected = spec["exclusions"]["expected_component_counts"][experiment_id]
        if len(sets["component_tokens"]) != expected:
            raise ContractError(
                f"selection manifest: exclusion-{experiment_id} component count mismatch"
            )
        runtime_components[experiment_id] = sets["component_tokens"]
    if runtime_components["670"] & runtime_components["672"]:
        raise ContractError("exclusion sources 670/672: component intersection must be zero")
    excluded = runtime_components["670"] | runtime_components["672"]
    if len(excluded) != spec["exclusions"]["expected_component_counts"]["union"]:
        raise ContractError("exclusion sources 670/672: component union mismatch")
    selected: set[str] = set()
    for index, binding in enumerate(manifest["selected_bindings"], 1):
        component = binding["component_token"]
        if component in excluded:
            raise ContractError(f"selection binding {index}: overlaps excluded component")
        if component in selected:
            raise ContractError("selection bindings: duplicate component token")
        selected.add(component)


def validate_review(review: Any, index: int, spec: dict[str, Any]) -> dict[str, Any]:
    context = f"review overlay row {index}"
    if not isinstance(review, dict):
        raise ContractError(f"{context}: review must be an object")
    expect_exact_keys(review, REVIEW_FIELDS, context)
    if any(type(review[field]) is not bool for field in BOOLEAN_REVIEW_FIELDS):
        raise ContractError(f"{context}: review flags must be booleans")
    if review["evidence_slice"] not in spec["enums"]["evidence_slice"]:
        raise ContractError(f"{context}: unknown evidence slice")
    return review


def _validate_times(contract: dict[str, Any], context: str) -> None:
    started = contract["started_at_utc"]
    completed = contract["completed_at_utc"]
    if not isinstance(started, str) or not TIMESTAMP.fullmatch(started):
        raise ContractError(f"{context}: invalid start timestamp")
    if not isinstance(completed, str) or not TIMESTAMP.fullmatch(completed) or completed < started:
        raise ContractError(f"{context}: invalid completion timestamp")


def validate_reviewer_contract(
    contract: dict[str, Any],
    *,
    role: str,
    overlay_path: Path,
    frozen_packet_path: Path,
    spec: dict[str, Any],
) -> str:
    context = f"reviewer {role} contract"
    expect_exact_keys(
        contract,
        {
            "schema_version",
            "target_audit_sha256",
            "rubric_sha256",
            "review_overlay_sha256",
            "reviewer_id_sha256",
            "started_at_utc",
            "completed_at_utc",
            "review_rows",
            "blind_to_other_reviewer",
            "blind_to_outcome_labels",
            "blind_to_prior_audits",
            "self_sha256",
        },
        context,
    )
    validate_self_hash(contract, context)
    if contract["schema_version"] != "exp689_reviewer_contract_v1":
        raise ContractError(f"{context}: schema version mismatch")
    if contract["target_audit_sha256"] != sha256_file(frozen_packet_path):
        raise ContractError(f"{context}: target packet lineage mismatch")
    if contract["rubric_sha256"] != spec["review_contract"]["rubric_sha256"]:
        raise ContractError(f"{context}: rubric lineage mismatch")
    if contract["review_overlay_sha256"] != sha256_file(overlay_path):
        raise ContractError(f"{context}: overlay SHA mismatch")
    reviewer_id = require_hex64(contract["reviewer_id_sha256"], f"{context}.reviewer_id")
    _validate_times(contract, context)
    if contract["review_rows"] != spec["review_contract"]["required_rows_per_reviewer"]:
        raise ContractError(f"{context}: review must contain exactly 300 rows")
    for field in (
        "blind_to_other_reviewer",
        "blind_to_outcome_labels",
        "blind_to_prior_audits",
    ):
        if contract[field] is not True:
            raise ContractError(f"{context}: {field} must be true")
    return reviewer_id


def load_review_overlay(
    path: Path, *, role: str, spec: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    rows = read_jsonl(path, f"reviewer {role} overlay")
    if len(rows) != spec["review_contract"]["required_rows_per_reviewer"]:
        raise ContractError(f"reviewer {role} overlay: expected exactly 300 rows")
    parsed: list[tuple[str, dict[str, Any]]] = []
    for index, row in enumerate(rows, 1):
        expect_exact_keys(
            row, {"schema_version", "audit_id", "review"}, f"reviewer {role} row"
        )
        if row["schema_version"] != "exp689_review_overlay_row_v1":
            raise ContractError(f"reviewer {role} row {index}: schema version mismatch")
        if row["audit_id"] != f"G689-{index:03d}":
            raise ContractError(f"reviewer {role} row {index}: audit order mismatch")
        parsed.append((row["audit_id"], validate_review(row["review"], index, spec)))
    return parsed


def validate_adjudication_contract(
    contract: dict[str, Any],
    *,
    overlay_path: Path,
    frozen_packet_path: Path,
    review_a_path: Path,
    review_b_path: Path,
    disagreement_ids: list[str],
    reviewer_ids: set[str],
    spec: dict[str, Any],
) -> None:
    context = "adjudication contract"
    expect_exact_keys(
        contract,
        {
            "schema_version",
            "target_audit_sha256",
            "rubric_sha256",
            "review_a_overlay_sha256",
            "review_b_overlay_sha256",
            "adjudication_overlay_sha256",
            "adjudicator_id_sha256",
            "started_at_utc",
            "completed_at_utc",
            "adjudication_rows",
            "disagreement_audit_ids_sha256",
            "blind_to_outcome_labels",
            "self_sha256",
        },
        context,
    )
    validate_self_hash(contract, context)
    if contract["schema_version"] != "exp689_adjudication_contract_v1":
        raise ContractError(f"{context}: schema version mismatch")
    exact_bindings = {
        "target_audit_sha256": sha256_file(frozen_packet_path),
        "rubric_sha256": spec["review_contract"]["rubric_sha256"],
        "review_a_overlay_sha256": sha256_file(review_a_path),
        "review_b_overlay_sha256": sha256_file(review_b_path),
        "adjudication_overlay_sha256": sha256_file(overlay_path),
        "disagreement_audit_ids_sha256": sha256_bytes(canonical_json_bytes(disagreement_ids)),
    }
    for field, expected in exact_bindings.items():
        if contract[field] != expected:
            raise ContractError(f"{context}: {field} lineage mismatch")
    adjudicator_id = require_hex64(
        contract["adjudicator_id_sha256"], f"{context}.adjudicator_id"
    )
    if adjudicator_id in reviewer_ids:
        raise ContractError("adjudicator must be distinct from both reviewers")
    _validate_times(contract, context)
    if contract["adjudication_rows"] != len(disagreement_ids):
        raise ContractError(f"{context}: disagreement row count mismatch")
    if contract["blind_to_outcome_labels"] is not True:
        raise ContractError(f"{context}: outcome-label blindness must be true")


def binary_agreement(left: list[bool], right: list[bool]) -> tuple[float, float, float]:
    if len(left) != len(right) or not left:
        raise ContractError("agreement inputs must have equal nonzero length")
    size = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / size
    left_true = sum(left) / size
    right_true = sum(right) / size
    expected_kappa = left_true * right_true + (1 - left_true) * (1 - right_true)
    kappa = (
        1.0
        if expected_kappa == 1.0 and observed == 1.0
        else (observed - expected_kappa) / (1 - expected_kappa)
    )
    pooled_true = (sum(left) + sum(right)) / (2 * size)
    expected_ac1 = 2 * pooled_true * (1 - pooled_true)
    ac1 = (
        1.0
        if expected_ac1 == 1.0 and observed == 1.0
        else (observed - expected_ac1) / (1 - expected_ac1)
    )
    return observed, kappa, ac1


def validate(
    *,
    remote_root: Path,
    source_rows_path: Path,
    source_contract_path: Path,
    selection_manifest_path: Path,
    teacher_request_path: Path,
    teacher_selections_path: Path,
    teacher_selection_contract_path: Path,
    frozen_packet_path: Path,
    review_a_path: Path,
    review_a_contract_path: Path,
    review_b_path: Path,
    review_b_contract_path: Path,
    adjudication_path: Path | None,
    adjudication_contract_path: Path | None,
    packet_contract_path: Path,
    exclusion_670_path: Path,
    exclusion_672_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate exact lineage, dual review, adjudication, and every frozen gate."""
    spec = load_spec()
    manifest, requests = load_prepared(
        remote_root=remote_root,
        selection_manifest_path=selection_manifest_path,
        teacher_request_path=teacher_request_path,
    )
    verify_prepared_against_sources(
        remote_root=remote_root,
        source_rows_path=source_rows_path,
        source_contract_path=source_contract_path,
        exclusion_670_path=exclusion_670_path,
        exclusion_672_path=exclusion_672_path,
        manifest=manifest,
        requests=requests,
    )
    selection_contract, selections = load_teacher_selections(
        remote_root=remote_root,
        selection_rows_path=teacher_selections_path,
        selection_contract_path=teacher_selection_contract_path,
        manifest=manifest,
        requests=requests,
    )
    required_paths = {
        "frozen audit packet": frozen_packet_path,
        "reviewer A overlay": review_a_path,
        "reviewer A contract": review_a_contract_path,
        "reviewer B overlay": review_b_path,
        "reviewer B contract": review_b_contract_path,
        "target-audit contract": packet_contract_path,
        "exclusion 670": exclusion_670_path,
        "exclusion 672": exclusion_672_path,
    }
    resolved = {
        name: require_remote_path(remote_root, path, context=name, must_exist=True)
        for name, path in required_paths.items()
    }
    output_path = require_remote_path(
        remote_root, output_path, context="audit result", must_exist=False
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable result: {output_path}")
    if (adjudication_path is None) != (adjudication_contract_path is None):
        raise ContractError("adjudication overlay and contract must be supplied together")
    resolved_adjudication: Path | None = None
    resolved_adjudication_contract: Path | None = None
    if adjudication_path is not None and adjudication_contract_path is not None:
        resolved_adjudication = require_remote_path(
            remote_root, adjudication_path, context="adjudication overlay", must_exist=True
        )
        resolved_adjudication_contract = require_remote_path(
            remote_root,
            adjudication_contract_path,
            context="adjudication contract",
            must_exist=True,
        )
    if len(
        {
            resolved["frozen audit packet"],
            resolved["reviewer A overlay"],
            resolved["reviewer B overlay"],
        }
    ) != 3:
        raise ContractError("frozen packet and reviewer overlays must be distinct files")

    packet_contract = load_json(resolved["target-audit contract"], "target-audit contract")
    validate_packet_contract(
        packet_contract,
        frozen_packet_path=resolved["frozen audit packet"],
        manifest=manifest,
        selection_contract=selection_contract,
        spec=spec,
    )
    validate_component_disjointness(
        manifest=manifest,
        exclusion_670_path=resolved["exclusion 670"],
        exclusion_672_path=resolved["exclusion 672"],
        spec=spec,
    )

    frozen_rows = read_jsonl(resolved["frozen audit packet"], "frozen audit packet")
    if len(frozen_rows) != spec["audit"]["total_rows"]:
        raise ContractError("frozen audit packet must contain exactly 300 rows")
    reviews_a = load_review_overlay(resolved["reviewer A overlay"], role="A", spec=spec)
    reviews_b = load_review_overlay(resolved["reviewer B overlay"], role="B", spec=spec)
    contract_a = load_json(resolved["reviewer A contract"], "reviewer A contract")
    contract_b = load_json(resolved["reviewer B contract"], "reviewer B contract")
    reviewer_a_id = validate_reviewer_contract(
        contract_a,
        role="A",
        overlay_path=resolved["reviewer A overlay"],
        frozen_packet_path=resolved["frozen audit packet"],
        spec=spec,
    )
    reviewer_b_id = validate_reviewer_contract(
        contract_b,
        role="B",
        overlay_path=resolved["reviewer B overlay"],
        frozen_packet_path=resolved["frozen audit packet"],
        spec=spec,
    )
    if reviewer_a_id == reviewer_b_id:
        raise ContractError("reviewer A and reviewer B must be distinct actors")

    for (audit_id_a, _), (audit_id_b, _) in zip(reviews_a, reviews_b, strict=True):
        if audit_id_a != audit_id_b:
            raise ContractError("review overlays: audit ID order mismatch")
    disagreement_ids = [
        audit_id_a
        for (audit_id_a, review_a), (_, review_b) in zip(reviews_a, reviews_b, strict=True)
        if review_a != review_b
    ]
    adjudicated: dict[str, dict[str, Any]] = {}
    adjudication_contract: dict[str, Any] | None = None
    if disagreement_ids:
        if resolved_adjudication is None or resolved_adjudication_contract is None:
            raise ContractError("every reviewer disagreement requires adjudication")
        adjudication_rows = read_jsonl(resolved_adjudication, "adjudication overlay")
        if len(adjudication_rows) != len(disagreement_ids):
            raise ContractError("adjudication must contain exactly every disagreement")
        for index, (row, expected_id) in enumerate(
            zip(adjudication_rows, disagreement_ids, strict=True), 1
        ):
            expect_exact_keys(
                row,
                {"schema_version", "audit_id", "review"},
                f"adjudication row {index}",
            )
            if row["schema_version"] != "exp689_adjudication_overlay_row_v1":
                raise ContractError(f"adjudication row {index}: schema version mismatch")
            if row["audit_id"] != expected_id:
                raise ContractError("adjudication is missing or includes a non-disagreement row")
            adjudicated[expected_id] = validate_review(row["review"], index, spec)
        adjudication_contract = load_json(
            resolved_adjudication_contract, "adjudication contract"
        )
        validate_adjudication_contract(
            adjudication_contract,
            overlay_path=resolved_adjudication,
            frozen_packet_path=resolved["frozen audit packet"],
            review_a_path=resolved["reviewer A overlay"],
            review_b_path=resolved["reviewer B overlay"],
            disagreement_ids=disagreement_ids,
            reviewer_ids={reviewer_a_id, reviewer_b_id},
            spec=spec,
        )
    elif resolved_adjudication is not None or resolved_adjudication_contract is not None:
        raise ContractError("adjudication is forbidden when reviewers have no disagreements")

    counts: Counter[str] = Counter()
    seen_record_ids: set[str] = set()
    seen_source_cards: set[str] = set()
    for index, (request, binding, selection, frozen, review_a_entry) in enumerate(
        zip(
            requests,
            manifest["selected_bindings"],
            selections,
            frozen_rows,
            reviews_a,
            strict=True,
        ),
        1,
    ):
        expected = make_target_row(request, binding, selection)
        if frozen != expected:
            raise ContractError(f"frozen audit row {index}: content differs from exact lineage")
        if review_a_entry[0] != frozen["audit_id"]:
            raise ContractError(f"review overlay row {index}: target packet binding mismatch")
        review = adjudicated.get(frozen["audit_id"], review_a_entry[1])
        record_id = frozen["record_id"]
        source_card = frozen["source_card_sha256"]
        if record_id in seen_record_ids or source_card in seen_source_cards:
            raise ContractError(f"frozen audit row {index}: exact duplicate detected")
        seen_record_ids.add(record_id)
        seen_source_cards.add(source_card)

        counts["schema_grounding"] += 1
        object_relation = review["sold_object_correct"] and review["relation_correct"]
        target_tuple = object_relation and review["substance_correct"]
        strict_pass = (
            target_tuple
            and review["evidence_supported"]
            and not review["unsupported_visual_claim"]
            and not review["contradiction"]
        )
        counts["sold_object_relation"] += int(object_relation)
        counts["target_tuple"] += int(target_tuple)
        stratum_name = frozen["stratum"]
        stratum_label = next(
            label
            for label, value in spec["review_contract"]["stratum_gates"].items()
            if value == stratum_name
        )
        counts[f"{stratum_label}_rows"] += 1
        counts[f"{stratum_label}_object_relation"] += int(object_relation)
        counts["evidence_supported"] += int(review["evidence_supported"])
        counts["contradictions"] += int(review["contradiction"])
        if review["evidence_slice"] == "image_required":
            counts["image_required"] += 1
            counts["image_required_strict_pass"] += int(strict_pass)
            counts["image_required_unsupported_visual"] += int(
                review["unsupported_visual_claim"]
            )
        if frozen["target"]["supervise"]:
            counts["supervised"] += 1
            if stratum_name == spec["audit"]["rare_positive_stratum"]:
                counts["rare_positive_supervised"] += 1
        if stratum_name == spec["audit"]["rare_positive_stratum"]:
            counts["rare_positive_rows"] += 1

    for label in ("S1", "S2", "S3"):
        if counts[f"{label}_rows"] != 100:
            raise ContractError(f"{label}: expected exactly 100 rows")
    if counts["rare_positive_rows"] != 100:
        raise ContractError("rare-positive stratum must contain exactly 100 rows")

    raw_exact_agreement = sum(
        review_a == review_b
        for (_, review_a), (_, review_b) in zip(reviews_a, reviews_b, strict=True)
    ) / 300
    per_attribute_agreement = {
        field: sum(
            review_a[field] == review_b[field]
            for (_, review_a), (_, review_b) in zip(reviews_a, reviews_b, strict=True)
        )
        / 300
        for field in sorted(REVIEW_FIELDS)
    }
    relation_a = [review["relation_correct"] for _, review in reviews_a]
    relation_b = [review["relation_correct"] for _, review in reviews_b]
    _, relation_kappa, relation_ac1 = binary_agreement(relation_a, relation_b)

    gates = spec["gates"]
    gate_checks = {
        "schema_grounding": counts["schema_grounding"] == gates["required_schema_grounding"],
        "target_tuple_correct": counts["target_tuple"] >= gates["minimum_target_tuple_correct"],
        "S1_object_relation_correct": (
            counts["S1_object_relation"] >= gates["minimum_s1_object_relation_correct"]
        ),
        "S2_object_relation_correct": (
            counts["S2_object_relation"] >= gates["minimum_s2_object_relation_correct"]
        ),
        "S3_object_relation_correct": (
            counts["S3_object_relation"] >= gates["minimum_s3_object_relation_correct"]
        ),
        "evidence_supported": counts["evidence_supported"] >= gates["minimum_evidence_supported"],
        "overall_coverage": counts["supervised"] >= gates["minimum_overall_supervised"],
        "rare_positive_coverage": (
            counts["rare_positive_supervised"] >= gates["minimum_rare_positive_supervised"]
        ),
        "contradiction_rate": counts["contradictions"] <= gates["maximum_contradictions"],
        "exact_review_agreement": raw_exact_agreement >= gates["minimum_exact_review_agreement"],
        "relation_cohen_kappa": relation_kappa >= gates["minimum_relation_cohen_kappa"],
        "image_required_rows": counts["image_required"] >= gates["minimum_image_required_rows"],
        "image_required_strict_pass": (
            counts["image_required_strict_pass"] >= gates["minimum_image_required_strict_pass"]
        ),
        "image_required_visual_support": (
            counts["image_required_unsupported_visual"]
            <= gates["maximum_image_required_unsupported_visual_claims"]
        ),
        "exact_nonduplication": (
            len(seen_record_ids) == gates["required_nonduplicate_rows"]
            and len(seen_source_cards) == gates["required_nonduplicate_rows"]
        ),
        "component_disjoint_from_670_672": True,
    }
    accepted = all(gate_checks.values())
    result = with_self_hash(
        {
            "schema_version": "exp689_target_audit_acceptance_v2",
            "experiment_id": "689",
            "status": "accepted" if accepted else "rejected_by_frozen_gate",
            "decision": "READY_FOR_SEPARATE_STUDENT_GPU_GO" if accepted else "NO_GO",
            "review_rows_per_reviewer": 300,
            "disagreement_rows": len(disagreement_ids),
            "metrics": {
                "schema_grounding": counts["schema_grounding"],
                "target_tuple_correct": counts["target_tuple"],
                "sold_object_relation_correct": counts["sold_object_relation"],
                "S1_object_relation_correct": counts["S1_object_relation"],
                "S2_object_relation_correct": counts["S2_object_relation"],
                "S3_object_relation_correct": counts["S3_object_relation"],
                "evidence_supported": counts["evidence_supported"],
                "unsupported_or_incorrect_evidence": 300 - counts["evidence_supported"],
                "supervised": counts["supervised"],
                "overall_coverage": counts["supervised"] / 300,
                "rare_positive_supervised": counts["rare_positive_supervised"],
                "rare_positive_coverage": counts["rare_positive_supervised"] / 100,
                "contradictions": counts["contradictions"],
                "contradiction_rate": counts["contradictions"] / 300,
                "image_required_rows": counts["image_required"],
                "image_required_strict_pass": counts["image_required_strict_pass"],
                "image_required_unsupported_visual_claims": counts[
                    "image_required_unsupported_visual"
                ],
                "raw_exact_review_agreement": raw_exact_agreement,
                "per_attribute_review_agreement": per_attribute_agreement,
                "relation_cohen_kappa": relation_kappa,
                "relation_gwet_ac1": relation_ac1,
                "unique_records": len(seen_record_ids),
                "unique_exact_sources": len(seen_source_cards),
            },
            "gate_checks": gate_checks,
            "frozen_spec_sha256": sha256_file(SPEC_PATH),
            "review_rubric_sha256": sha256_file(RUBRIC_PATH),
            "selection_manifest_self_sha256": manifest["self_sha256"],
            "target_audit_contract_self_sha256": packet_contract["self_sha256"],
            "frozen_packet_sha256": sha256_file(resolved["frozen audit packet"]),
            "review_a_contract_self_sha256": contract_a["self_sha256"],
            "review_b_contract_self_sha256": contract_b["self_sha256"],
            "review_a_overlay_sha256": sha256_file(resolved["reviewer A overlay"]),
            "review_b_overlay_sha256": sha256_file(resolved["reviewer B overlay"]),
            "adjudication_contract_self_sha256": (
                adjudication_contract["self_sha256"] if adjudication_contract else None
            ),
            "student_gpu_authorized": False,
            "fresh_independent_gpu_go_required": True,
            "jobs_launched": 0,
            "uploads": 0,
            "presets_built": 0,
            "bundles_built": 0,
            "public_used": False,
            "self_sha256": None,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--teacher-request", type=Path, required=True)
    parser.add_argument("--teacher-selections", type=Path, required=True)
    parser.add_argument("--teacher-selection-contract", type=Path, required=True)
    parser.add_argument("--frozen-packet", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-a-contract", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--review-b-contract", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--adjudication-contract", type=Path)
    parser.add_argument("--packet-contract", type=Path, required=True)
    parser.add_argument("--exclusion-670", type=Path, required=True)
    parser.add_argument("--exclusion-672", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(
        remote_root=args.remote_root,
        source_rows_path=args.source_rows,
        source_contract_path=args.source_contract,
        selection_manifest_path=args.selection_manifest,
        teacher_request_path=args.teacher_request,
        teacher_selections_path=args.teacher_selections,
        teacher_selection_contract_path=args.teacher_selection_contract,
        frozen_packet_path=args.frozen_packet,
        review_a_path=args.review_a,
        review_a_contract_path=args.review_a_contract,
        review_b_path=args.review_b,
        review_b_contract_path=args.review_b_contract,
        adjudication_path=args.adjudication,
        adjudication_contract_path=args.adjudication_contract,
        packet_contract_path=args.packet_contract,
        exclusion_670_path=args.exclusion_670,
        exclusion_672_path=args.exclusion_672,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
