from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "686"
AUTHORITY = "independent_main_integrator"
HEX64_FIELDS = (
    "evaluation_sha256",
    "evaluation_file_sha256",
    "source_output_metadata_sha256",
    "evaluation_code_bundle_sha256",
    "evaluation_code_acceptance_sha256",
    "replay_bundle_sha256",
    "replay_contract_sha256",
    "registry_sha256",
)
EVIDENCE_FIELDS = (
    "candidate_acceptance_sha256",
    "control_acceptance_sha256",
    "candidate_prediction_sha256",
    "control_prediction_sha256",
)
REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "stage",
    "folds",
    "passed",
    "decision",
    "gates",
    *HEX64_FIELDS,
    *EVIDENCE_FIELDS,
    "evaluation_code_revision",
    "evaluator_job_id",
    "evaluator_job_state",
    "evaluation_output_source",
    "gate_authority",
    "gate_main_revision",
    "parent_promotion_gate_sha256",
    "parent_promotion_gate_file_sha256",
    "parent_promotion_evaluation_file_sha256",
    "parent_promotion_source",
    "validation_labels_read_by_training",
    "sealed_rows",
    "public_used",
    "promotion_gate_sha256",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_hex(value: Any, size: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == size
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_hash_evidence(value: Any) -> bool:
    if is_hex(value, 64):
        return True
    if isinstance(value, list) and value:
        return all(validate_hash_evidence(item) for item in value)
    if isinstance(value, dict) and value:
        return all(
            isinstance(key, str) and key and validate_hash_evidence(item)
            for key, item in value.items()
        )
    return False


def verify_promotion_gate(
    gate_path: Path,
    evaluation_path: Path,
    *,
    expected_stage: str,
    expected_folds: list[int],
    expected_decision: str,
    expected_source: str | None = None,
) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if set(gate) != REQUIRED_FIELDS:
        raise ValueError("promotion gate schema mismatch")
    gate_body = dict(gate)
    gate_digest = gate_body.pop("promotion_gate_sha256")
    if gate_digest != canonical_sha256(gate_body):
        raise ValueError("promotion gate self-hash mismatch")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation_body = dict(evaluation)
    evaluation_digest = evaluation_body.pop("evaluation_sha256", None)
    if evaluation_digest != canonical_sha256(evaluation_body):
        raise ValueError("scientific evaluation self-hash mismatch")
    raw_expected = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": expected_stage,
        "folds": expected_folds,
        "passed": True,
        "decision": expected_decision,
        "validation_labels_read_by_training": 0,
        "sealed_rows": 0,
        "public_used": False,
    }
    if any(evaluation.get(key) != value for key, value in raw_expected.items()):
        raise ValueError("raw scientific evaluation does not authorize this stage")
    expected = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": expected_stage,
        "folds": expected_folds,
        "passed": True,
        "decision": expected_decision,
        "evaluation_sha256": evaluation_digest,
        "evaluation_file_sha256": sha256_file(evaluation_path),
        "evaluator_job_state": "SUCCEEDED",
        "gate_authority": AUTHORITY,
        "validation_labels_read_by_training": 0,
        "sealed_rows": 0,
        "public_used": False,
    }
    if expected_source is not None:
        expected["evaluation_output_source"] = expected_source
    if any(gate.get(key) != value for key, value in expected.items()):
        raise ValueError("promotion gate identity or scientific decision mismatch")
    for field in HEX64_FIELDS:
        if not is_hex(gate.get(field), 64):
            raise ValueError("promotion gate hash provenance is invalid")
    for field in EVIDENCE_FIELDS:
        if not validate_hash_evidence(gate.get(field)):
            raise ValueError("promotion gate input evidence is invalid")
    for field in (
        "gates",
        "evaluation_code_bundle_sha256",
        "evaluation_code_revision",
        "evaluation_code_acceptance_sha256",
        "candidate_acceptance_sha256",
        "control_acceptance_sha256",
        "candidate_prediction_sha256",
        "control_prediction_sha256",
        "replay_bundle_sha256",
        "replay_contract_sha256",
        "registry_sha256",
    ):
        if gate.get(field) != evaluation.get(field):
            raise ValueError("promotion gate differs from scientific evaluation evidence")
    gates = gate["gates"]
    if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
        raise ValueError("promotion gate contains a failed or malformed scientific gate")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(gate["evaluator_job_id"])):
        raise ValueError("evaluator job identity is invalid")
    if not is_hex(gate["gate_main_revision"], 40) or not is_hex(
        gate["evaluation_code_revision"], 40
    ):
        raise ValueError("promotion gate revision is invalid")
    if not isinstance(gate["evaluation_output_source"], str) or not gate[
        "evaluation_output_source"
    ]:
        raise ValueError("promotion output source is invalid")
    if expected_stage == "blind3":
        expected_parent = (None, None, None, None)
    else:
        expected_parent = tuple(
            gate[field]
            for field in (
                "parent_promotion_gate_sha256",
                "parent_promotion_gate_file_sha256",
                "parent_promotion_evaluation_file_sha256",
                "parent_promotion_source",
            )
        )
        if not (
            is_hex(expected_parent[0], 64)
            and is_hex(expected_parent[1], 64)
            and is_hex(expected_parent[2], 64)
            and isinstance(expected_parent[3], str)
            and expected_parent[3]
        ):
            raise ValueError("promotion gate lacks parent-stage lineage")
    if expected_stage == "blind3" and expected_parent != (None, None, None, None):
        raise ValueError("blind3 promotion gate must not have a parent")
    evaluation_parent = tuple(
        evaluation.get(field)
        for field in (
            "promotion_receipt_sha256",
            "promotion_receipt_file_sha256",
            "promotion_evaluation_file_sha256",
            "promotion_receipt_source",
        )
    )
    if evaluation_parent != expected_parent:
        raise ValueError("promotion parent differs from raw scientific evaluation")
    return gate
