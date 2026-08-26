from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

TEACHER_EXPERIMENT_ID = "691"
ROUTED_EXPERIMENT_ID = "692"
FOLD_SCHEMA = "exp691_fold_report_v1"
ACCEPTANCE_SCHEMA = "exp692_qwen27_routed_acceptance_v1"
FLAMMABLE = "Легковоспламеняющиеся"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    body = dict(value)
    digest = body.pop(field, None)
    if digest != canonical_sha256(body):
        raise ValueError(f"{field} mismatch")


def verify_routed_acceptance(path: Path, *, expected_file_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("exp692 routed acceptance file SHA-256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    verify_self_hash(value, "acceptance_sha256")
    expected = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "experiment_id": ROUTED_EXPERIMENT_ID,
        "teacher_experiment_id": TEACHER_EXPERIMENT_ID,
        "teacher_scope": "all",
        "runtime_labels_read_after_teacher_terminal": True,
        "public_used": False,
        "sealed_rows": 0,
        "decision": "OPEN_THREE_STUDENT_METHODS",
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("exp692 routed acceptance contract mismatch")
    if set(value.get("folds", {})) != {str(fold) for fold in range(5)} or not isinstance(
        value.get("pooled"), dict
    ):
        raise ValueError("exp692 routed metrics are incomplete")
    gate = value.get("gate", {})
    if not (
        int(gate.get("fold_wins", -1)) >= 4
        and float(gate.get("flammable_ap_delta", 0.0)) > 0.0
        and float(gate.get("flammable_f1_delta", 0.0)) > 0.0
        and gate.get("flammable_fn_nonincrease") is True
        and gate.get("bad_byte_identical") is True
    ):
        raise ValueError("exp692 routed teacher quality gate is not open")
    bindings = value.get("artifact_bindings")
    if not isinstance(bindings, list) or [item.get("fold") for item in bindings] != list(range(5)):
        raise ValueError("exp692 artifact bindings do not cover ordered folds 0..4")
    report_hashes = value.get("teacher_fold_report_self_sha256")
    if not isinstance(report_hashes, list) or len(report_hashes) != 5:
        raise ValueError("exp692 fold report bindings are incomplete")
    for fold, binding in enumerate(bindings):
        if binding.get("fold_report_self_sha256") != report_hashes[fold]:
            raise ValueError("exp692 fold report binding mismatch")
        for field in (
            "fold_report_file_sha256",
            "fold_report_self_sha256",
            "predictions_sha256",
            "teacher_targets_sha256",
            "evidence_sha256",
        ):
            digest = binding.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"exp692 {field} is invalid")
        if not isinstance(binding.get("evidence_rows"), int) or binding["evidence_rows"] < 0:
            raise ValueError("exp692 evidence row count is invalid")
    return value


def _verify_fold_report(
    directory: Path,
    *,
    fold: int,
    runtime_contract_sha256: str,
    binding: dict[str, Any],
) -> dict[str, Any]:
    path = directory / "report.json"
    if sha256_file(path) != binding["fold_report_file_sha256"]:
        raise ValueError(f"exp691 fold {fold} report file SHA-256 mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    verify_self_hash(report, "report_sha256")
    expected = {
        "schema_version": FOLD_SCHEMA,
        "experiment_id": TEACHER_EXPERIMENT_ID,
        "scope": "all",
        "fold": fold,
        "runtime_contract_sha256": runtime_contract_sha256,
        "validation_labels_loaded_after_adapter_frozen": True,
        "public_used": False,
        "sealed_rows": 0,
        "decision": "FOLD_COMPLETE",
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"exp691 fold {fold} report contract mismatch")
    if report["report_sha256"] != binding["fold_report_self_sha256"]:
        raise ValueError(f"exp691 fold {fold} report differs from exp692 acceptance")
    return report


def _merge_targets(
    path: Path,
    *,
    report: dict[str, Any],
    binding: dict[str, Any],
    train: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload_sha = sha256_file(path)
    if (
        report.get("teacher_targets_sha256") != payload_sha
        or binding["teacher_targets_sha256"] != payload_sha
    ):
        raise ValueError("exp691 teacher_targets SHA-256 mismatch")
    targets = read_jsonl(path)
    if len(targets) != len(train) or report.get("teacher_target_rows") != len(targets):
        raise ValueError("exp691 teacher target row count mismatch")
    enriched: list[dict[str, Any]] = []
    for occurrence_index, (row, target) in enumerate(zip(train, targets, strict=True)):
        expected = {
            "global_index": int(row["global_index"]),
            "id": str(row["id"]),
            "fold": int(row["fold"]),
            "category": str(row["category"]),
            "occurrence_index": occurrence_index,
        }
        if set(target) != {*expected, "score"} or any(
            target.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("exp691 teacher target occurrence binding mismatch")
        score = float(target["score"])
        if not math.isfinite(score):
            raise ValueError("exp691 teacher target is non-finite")
        item = dict(row)
        if str(row["category"]) == FLAMMABLE:
            item["teacher_score"] = score
        enriched.append(item)
    return enriched


def _merge_evidence(
    path: Path,
    *,
    report: dict[str, Any],
    binding: dict[str, Any],
    train: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if sha256_file(path) != binding["evidence_sha256"]:
        raise ValueError("exp691 evidence SHA-256 mismatch")
    evidence_rows = read_jsonl(path)
    if report.get("evidence_rows") != len(evidence_rows) or binding["evidence_rows"] != len(
        evidence_rows
    ):
        raise ValueError("exp691 evidence row count mismatch")
    by_index: dict[int, dict[str, Any]] = {}
    for row in evidence_rows:
        key = int(row["global_index"])
        if key in by_index:
            raise ValueError("exp691 evidence has duplicate global_index")
        by_index[key] = row
    enriched: list[dict[str, Any]] = []
    for row in train:
        item = dict(row)
        if str(row["category"]) == FLAMMABLE:
            evidence_row = by_index.get(int(row["global_index"]))
            if evidence_row is None or str(evidence_row.get("id")) != str(row["id"]):
                raise ValueError("exp691 evidence identity binding mismatch")
            evidence = evidence_row.get("evidence")
            if not isinstance(evidence, dict):
                raise TypeError("exp691 evidence payload must be an object")
            item["teacher_evidence"] = evidence
        enriched.append(item)
    return enriched


def load_fold(
    teacher_root: Path,
    *,
    fold: int,
    train: list[dict[str, Any]],
    runtime_contract_sha256: str,
    require_evidence: bool,
    acceptance_path: Path,
    expected_acceptance_file_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    acceptance = verify_routed_acceptance(
        acceptance_path, expected_file_sha256=expected_acceptance_file_sha256
    )
    binding = acceptance["artifact_bindings"][fold]
    directory = teacher_root / f"fold{fold}"
    report = _verify_fold_report(
        directory,
        fold=fold,
        runtime_contract_sha256=runtime_contract_sha256,
        binding=binding,
    )
    enriched = _merge_targets(
        directory / "teacher_targets.jsonl",
        report=report,
        binding=binding,
        train=train,
    )
    if require_evidence:
        enriched = _merge_evidence(
            directory / "evidence.jsonl",
            report=report,
            binding=binding,
            train=enriched,
        )
    consumer_binding = {
        "teacher_experiment_id": TEACHER_EXPERIMENT_ID,
        "routed_acceptance_experiment_id": ROUTED_EXPERIMENT_ID,
        "fold": fold,
        "fold_report_file_sha256": binding["fold_report_file_sha256"],
        "fold_report_self_sha256": binding["fold_report_self_sha256"],
        "teacher_targets_sha256": binding["teacher_targets_sha256"],
        "evidence_sha256": binding["evidence_sha256"] if require_evidence else None,
        "routed_acceptance_sha256": acceptance["acceptance_sha256"],
        "routed_acceptance_file_sha256": expected_acceptance_file_sha256,
        "scope": "all",
    }
    return enriched, consumer_binding
