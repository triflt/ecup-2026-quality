"""Validate a completed experiment-689 audit and emit a non-authorizing gate result."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_target_audit import (
    SPEC_PATH,
    ContractError,
    load_json,
    load_prepared,
    load_runtime_exclusion,
    load_spec,
    load_teacher_selections,
    make_target_row,
    read_jsonl,
    require_remote_path,
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
    "contradiction",
}


def validate_packet_contract(
    contract: dict[str, Any],
    *,
    frozen_packet_path: Path,
    manifest: dict[str, Any],
    selection_contract: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    expected_fields = {
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
        "human_review_fields_empty",
        "student_gpu_authorized",
        "self_sha256",
    }
    actual_fields = set(contract)
    if actual_fields != expected_fields:
        raise ContractError("target-audit contract: exact fields required")
    validate_self_hash(contract, "target-audit contract")
    if contract["schema_version"] != "exp689_target_audit_contract_v1":
        raise ContractError("target-audit contract: schema version mismatch")
    if contract["experiment_id"] != "689" or contract["execution_scope"] != "remote_remote_compute":
        raise ContractError("target-audit contract: experiment or execution scope mismatch")
    if contract["frozen_spec_sha256"] != sha256_file(SPEC_PATH):
        raise ContractError("target-audit contract: frozen spec SHA mismatch")
    if contract["target_audit_schema_sha256"] != spec["target_audit_schema_sha256"]:
        raise ContractError("target-audit contract: frozen schema SHA mismatch")
    if contract["selection_manifest_self_sha256"] != manifest["self_sha256"]:
        raise ContractError("target-audit contract: selection manifest lineage mismatch")
    if contract["teacher_request_sha256"] != manifest["teacher_request_sha256"]:
        raise ContractError("target-audit contract: teacher request lineage mismatch")
    if (
        contract["teacher_selection_contract_self_sha256"]
        != selection_contract["self_sha256"]
    ):
        raise ContractError("target-audit contract: teacher selection contract lineage mismatch")
    if contract["teacher_selections_sha256"] != selection_contract["selection_rows_sha256"]:
        raise ContractError("target-audit contract: teacher selection rows lineage mismatch")
    if contract["target_audit_sha256"] != sha256_file(frozen_packet_path):
        raise ContractError("target-audit contract: frozen packet SHA mismatch")
    if contract["target_audit_rows"] != spec["audit"]["total_rows"]:
        raise ContractError("target-audit contract: row count mismatch")
    if contract["selected_counts_by_stratum"] != manifest["selected_counts_by_stratum"]:
        raise ContractError("target-audit contract: stratum counts mismatch")
    if contract["gates"] != spec["gates"]:
        raise ContractError("target-audit contract: gates differ from frozen spec")
    if contract["human_review_fields_empty"] is not True:
        raise ContractError("target-audit contract: frozen review fields must be empty")
    if contract["student_gpu_authorized"] is not False:
        raise ContractError("target-audit contract: must not authorize student GPU")


def validate_disjointness(
    *,
    manifest: dict[str, Any],
    exclusion_670_path: Path,
    exclusion_672_path: Path,
    spec: dict[str, Any],
) -> None:
    runtime_sets: dict[str, dict[str, set[str]]] = {}
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
        expected_count = spec["exclusions"]["expected_component_counts"][experiment_id]
        if len(sets["component_tokens"]) != expected_count:
            raise ContractError(
                f"selection manifest: exclusion-{experiment_id} component count mismatch"
            )
        runtime_sets[experiment_id] = sets
    sets_670 = runtime_sets["670"]
    sets_672 = runtime_sets["672"]
    if (
        len(sets_670["component_tokens"] & sets_672["component_tokens"])
        != spec["exclusions"]["required_component_intersection"]
    ):
        raise ContractError("exclusion sources 670/672: component intersection must be zero")
    if (
        len(sets_670["component_tokens"] | sets_672["component_tokens"])
        != spec["exclusions"]["expected_component_counts"]["union"]
    ):
        raise ContractError("exclusion sources 670/672: component union mismatch")
    excluded = {
        field: sets_670[field] | sets_672[field]
        for field in ("row_tokens", "component_tokens", "family_tokens")
    }
    row_tokens: set[str] = set()
    component_tokens: set[str] = set()
    source_cards: set[str] = set()
    for index, binding in enumerate(manifest["selected_bindings"], 1):
        if binding["row_token"] in excluded["row_tokens"]:
            raise ContractError(f"selection binding {index}: overlaps excluded audit row")
        if binding["component_token"] in excluded["component_tokens"]:
            raise ContractError(f"selection binding {index}: overlaps excluded component")
        if binding["family_token"] in excluded["family_tokens"]:
            raise ContractError(f"selection binding {index}: overlaps excluded family")
        if binding["row_token"] in row_tokens:
            raise ContractError("selection bindings: duplicate row token")
        if binding["component_token"] in component_tokens:
            raise ContractError("selection bindings: duplicate component token")
        if binding["source_card_sha256"] in source_cards:
            raise ContractError("selection bindings: duplicate exact source card")
        row_tokens.add(binding["row_token"])
        component_tokens.add(binding["component_token"])
        source_cards.add(binding["source_card_sha256"])


def validate_completed_review(review: Any, index: int) -> dict[str, bool]:
    if not isinstance(review, dict) or set(review) != REVIEW_FIELDS:
        raise ContractError(f"completed review row {index}: exact review fields required")
    if any(type(review[field]) is not bool for field in REVIEW_FIELDS):
        raise ContractError(f"completed review row {index}: every review field must be boolean")
    return review


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
    completed_review_path: Path,
    packet_contract_path: Path,
    exclusion_670_path: Path,
    exclusion_672_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate immutable lineage, manual ratings, and every preregistered gate."""
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
    frozen_packet_path = require_remote_path(
        remote_root, frozen_packet_path, context="frozen audit packet", must_exist=True
    )
    completed_review_path = require_remote_path(
        remote_root, completed_review_path, context="completed audit review", must_exist=True
    )
    packet_contract_path = require_remote_path(
        remote_root, packet_contract_path, context="target-audit contract", must_exist=True
    )
    exclusion_670_path = require_remote_path(
        remote_root, exclusion_670_path, context="exclusion 670", must_exist=True
    )
    exclusion_672_path = require_remote_path(
        remote_root, exclusion_672_path, context="exclusion 672", must_exist=True
    )
    output_path = require_remote_path(
        remote_root, output_path, context="audit result", must_exist=False
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable result: {output_path}")
    if frozen_packet_path == completed_review_path:
        raise ContractError("completed review must be a separate copy of the frozen packet")

    packet_contract = load_json(packet_contract_path, "target-audit contract")
    validate_packet_contract(
        packet_contract,
        frozen_packet_path=frozen_packet_path,
        manifest=manifest,
        selection_contract=selection_contract,
        spec=spec,
    )
    validate_disjointness(
        manifest=manifest,
        exclusion_670_path=exclusion_670_path,
        exclusion_672_path=exclusion_672_path,
        spec=spec,
    )

    frozen_rows = read_jsonl(frozen_packet_path, "frozen audit packet")
    completed_rows = read_jsonl(completed_review_path, "completed audit review")
    if len(frozen_rows) != 300 or len(completed_rows) != 300:
        raise ContractError("frozen and completed audit packets must each contain exactly 300 rows")

    counts: Counter[str] = Counter()
    seen_record_ids: set[str] = set()
    seen_source_cards: set[str] = set()
    for index, (request, binding, selection, frozen, completed) in enumerate(
        zip(
            requests,
            manifest["selected_bindings"],
            selections,
            frozen_rows,
            completed_rows,
            strict=True,
        ),
        1,
    ):
        expected = make_target_row(request, binding, selection)
        if frozen != expected:
            raise ContractError(f"frozen audit row {index}: content differs from exact lineage")
        if not isinstance(completed, dict) or set(completed) != set(frozen):
            raise ContractError(f"completed audit row {index}: exact packet fields required")
        for field in frozen:
            if field != "review" and completed[field] != frozen[field]:
                raise ContractError(f"completed audit row {index}: immutable field {field} changed")
        review = validate_completed_review(completed["review"], index)
        record_id = frozen["record_id"]
        source_card = frozen["source_card_sha256"]
        if record_id in seen_record_ids or source_card in seen_source_cards:
            raise ContractError(f"frozen audit row {index}: exact duplicate detected")
        seen_record_ids.add(record_id)
        seen_source_cards.add(source_card)

        counts["schema_grounding"] += 1
        sold_object_relation = review["sold_object_correct"] and review["relation_correct"]
        target_tuple = sold_object_relation and review["substance_correct"]
        counts["sold_object_relation"] += int(sold_object_relation)
        counts["target_tuple"] += int(target_tuple)
        if frozen["stratum"] == spec["audit"]["critical_stratum"]:
            counts["critical_rows"] += 1
            counts["critical_joint"] += int(sold_object_relation)
        counts["evidence_supported"] += int(review["evidence_supported"])
        counts["contradictions"] += int(review["contradiction"])
        if frozen["target"]["supervise"]:
            counts["supervised"] += 1
            if frozen["stratum"] == spec["audit"]["rare_positive_stratum"]:
                counts["rare_positive_supervised"] += 1
        if frozen["stratum"] == spec["audit"]["rare_positive_stratum"]:
            counts["rare_positive_rows"] += 1

    if counts["critical_rows"] != 100 or counts["rare_positive_rows"] != 100:
        raise ContractError("critical and rare-positive strata must each contain exactly 100 rows")
    gates = spec["gates"]
    gate_checks = {
        "schema_grounding": counts["schema_grounding"] == gates["required_schema_grounding"],
        "target_tuple_correct": (
            counts["target_tuple"] >= gates["minimum_target_tuple_correct"]
        ),
        "critical_joint_sold_object_relation": (
            counts["critical_joint"]
            >= gates["minimum_critical_joint_sold_object_relation"]
        ),
        "evidence_supported": (
            counts["evidence_supported"] >= gates["minimum_evidence_supported"]
        ),
        "overall_coverage": counts["supervised"] >= gates["minimum_overall_supervised"],
        "rare_positive_coverage": (
            counts["rare_positive_supervised"]
            >= gates["minimum_rare_positive_supervised"]
        ),
        "contradiction_rate": counts["contradictions"] <= gates["maximum_contradictions"],
        "exact_nonduplication": (
            len(seen_record_ids) == gates["required_nonduplicate_rows"]
            and len(seen_source_cards) == gates["required_nonduplicate_rows"]
        ),
        "component_disjoint_from_670_672": True,
    }
    accepted = all(gate_checks.values())
    result = with_self_hash(
        {
            "schema_version": "exp689_target_audit_acceptance_v1",
            "experiment_id": "689",
            "status": "accepted" if accepted else "rejected_by_frozen_gate",
            "decision": "READY_FOR_SEPARATE_STUDENT_GPU_GO" if accepted else "NO_GO",
            "review_rows": len(completed_rows),
            "metrics": {
                "schema_grounding": counts["schema_grounding"],
                "target_tuple_correct": counts["target_tuple"],
                "sold_object_relation_correct": counts["sold_object_relation"],
                "critical_joint_sold_object_relation": counts["critical_joint"],
                "evidence_supported": counts["evidence_supported"],
                "unsupported_or_incorrect_evidence": 300 - counts["evidence_supported"],
                "supervised": counts["supervised"],
                "overall_coverage": counts["supervised"] / 300,
                "rare_positive_supervised": counts["rare_positive_supervised"],
                "rare_positive_coverage": counts["rare_positive_supervised"] / 100,
                "contradictions": counts["contradictions"],
                "contradiction_rate": counts["contradictions"] / 300,
                "unique_records": len(seen_record_ids),
                "unique_exact_sources": len(seen_source_cards),
            },
            "gate_checks": gate_checks,
            "frozen_spec_sha256": sha256_file(SPEC_PATH),
            "selection_manifest_self_sha256": manifest["self_sha256"],
            "target_audit_contract_self_sha256": packet_contract["self_sha256"],
            "frozen_packet_sha256": sha256_file(frozen_packet_path),
            "completed_review_sha256": sha256_file(completed_review_path),
            "student_gpu_authorized": False,
            "fresh_independent_gpu_go_required": True,
            "jobs_launched": 0,
            "uploads": 0,
            "presets_built": 0,
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
    parser.add_argument("--completed-review", type=Path, required=True)
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
        completed_review_path=args.completed_review,
        packet_contract_path=args.packet_contract,
        exclusion_670_path=args.exclusion_670,
        exclusion_672_path=args.exclusion_672,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
