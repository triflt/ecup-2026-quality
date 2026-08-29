"""Diagnose the frozen fold0 validation binding without reading validation rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
RUNNER_PATH = f"{EXPERIMENT}/diagnose_fold0_validation_binding.py"
EXTRACTOR_PATH = f"{EXPERIMENT}/extract_source_archive_transport.py"
SPEC_PATH = f"{EXPERIMENT}/source_prepare_spec_v1.json"
BUNDLE_PATHS = {RUNNER_PATH, EXTRACTOR_PATH, SPEC_PATH}
BUNDLE_SCHEMA = "exp689_fold0_validation_diagnostic_bundle_manifest_v1"
REPORT_SCHEMA = "exp689_fold0_validation_binding_diagnostic_v1"
RUNTIME_ROOT = Path(
    "experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold0"
)
AUDIT_MEMBER = (RUNTIME_ROOT / "runtime_audit.json").as_posix()
VALIDATION_MEMBER = (RUNTIME_ROOT / "validation.jsonl").as_posix()
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
sys.dont_write_bytecode = True


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, context: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp689_fold0_transport", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load accepted transport extractor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _self_hash(value: dict[str, Any], *, field: str = "self_sha256") -> str:
    copy = dict(value)
    actual = copy.get(field)
    copy[field] = None
    recomputed = hashlib.sha256(canonical_json_bytes(copy)).hexdigest()
    if not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise ValueError(f"invalid {field}")
    if actual != recomputed:
        raise ValueError(f"canonical {field} mismatch")
    return actual


def validate_bundle(
    root: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_manifest_self_sha256: str,
    expected_revision: str,
) -> dict[str, Any]:
    if not HEX40.fullmatch(expected_revision) or not all(
        HEX64.fullmatch(value)
        for value in (expected_manifest_sha256, expected_manifest_self_sha256)
    ):
        raise ValueError("diagnostic bundle requires exact revision/hashes")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("diagnostic bundle manifest file SHA mismatch")
    manifest = _load_json(manifest_path, "diagnostic bundle manifest")
    if set(manifest) != {"schema_version", "builder_revision", "files", "self_sha256"}:
        raise ValueError("diagnostic bundle manifest exact schema mismatch")
    if _self_hash(manifest) != expected_manifest_self_sha256:
        raise ValueError("diagnostic bundle manifest self SHA mismatch")
    files = manifest["files"]
    if (
        manifest["schema_version"] != BUNDLE_SCHEMA
        or manifest["builder_revision"] != expected_revision
        or not isinstance(files, list)
        or [item.get("path") for item in files] != sorted(BUNDLE_PATHS)
    ):
        raise ValueError("diagnostic bundle exact whitelist mismatch")
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_paths != BUNDLE_PATHS:
        raise ValueError("diagnostic bundle contains non-whitelisted files")
    for item in files:
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("diagnostic bundle member schema mismatch")
        path = root / item["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError("diagnostic bundle member mismatch")
    return manifest


def _canonical_field(value: dict[str, Any], field: str, *, remove: bool) -> dict[str, Any]:
    declared = value.get(field)
    if declared is None:
        return {"present": False, "declared": None, "recomputed": None, "valid": None}
    copy = dict(value)
    if remove:
        copy.pop(field, None)
    else:
        copy[field] = None
    recomputed = hashlib.sha256(canonical_json_bytes(copy)).hexdigest()
    return {
        "present": True,
        "declared": declared,
        "recomputed": recomputed,
        "valid": isinstance(declared, str)
        and bool(HEX64.fullmatch(declared))
        and declared == recomputed,
    }


def classify(equality: dict[str, bool | None]) -> str:
    if equality["runtime_audit_contract_valid"] is not True:
        return "RUNTIME_AUDIT_CONTRACT_INVALID"
    if equality["source_runtime_contract_equals_frozen"] is not True:
        return "SOURCE_RUNTIME_CONTRACT_MISMATCH"
    declared_actual = equality["declared_equals_actual"]
    declared_frozen = equality["declared_equals_frozen"]
    actual_frozen = equality["actual_equals_frozen"]
    if declared_actual and declared_frozen and actual_frozen:
        return "ALL_VALIDATION_BINDINGS_EQUAL"
    if declared_frozen and not actual_frozen:
        return "ACTUAL_VALIDATION_DIFFERS_FROM_DECLARED_AND_FROZEN"
    if actual_frozen and not declared_frozen:
        return "DECLARED_VALIDATION_DIFFERS_FROM_ACTUAL_AND_FROZEN"
    if declared_actual and not actual_frozen:
        return "FROZEN_VALIDATION_EXPECTATION_DIFFERS"
    return "VALIDATION_BINDINGS_ALL_DIFFERENT_OR_MISSING"


def inspect_extracted(
    extraction_root: Path,
    *,
    spec: dict[str, Any],
) -> dict[str, Any]:
    audit_path = extraction_root / AUDIT_MEMBER
    validation_path = extraction_root / VALIDATION_MEMBER
    for path, context in (
        (audit_path, "fold0 runtime_audit"),
        (validation_path, "fold0 validation"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"exact {context} path is missing or substituted")
    audit = _load_json(audit_path, "fold0 runtime_audit")
    outputs = audit.get("output_sha256")
    declared_validation = (
        outputs.get("validation.jsonl") if isinstance(outputs, dict) else None
    )
    frozen = spec["folds"]["0"]
    actual_validation = sha256_file(validation_path)
    legacy_contract = _canonical_field(audit, "contract_sha256", remove=True)
    optional_self = _canonical_field(audit, "self_sha256", remove=False)
    equality: dict[str, bool | None] = {
        "declared_equals_actual": declared_validation == actual_validation,
        "declared_equals_frozen": declared_validation == frozen["validation_sha256"],
        "actual_equals_frozen": actual_validation == frozen["validation_sha256"],
        "source_runtime_contract_equals_frozen": (
            audit.get("source_runtime_contract_sha256")
            == frozen["source_runtime_contract_sha256"]
        ),
        "runtime_audit_contract_valid": legacy_contract["valid"],
        "runtime_audit_self_valid": optional_self["valid"],
    }
    return {
        "paths": {
            "runtime_audit": {
                "archive_member": AUDIT_MEMBER,
                "extracted_relative_path": AUDIT_MEMBER,
                "classification": "regular_file_runtime_metadata_json",
            },
            "validation": {
                "archive_member": VALIDATION_MEMBER,
                "extracted_relative_path": VALIDATION_MEMBER,
                "classification": "regular_file_opaque_validation_payload",
            },
        },
        "runtime_audit": {
            "file_sha256": sha256_file(audit_path),
            "size_bytes": audit_path.stat().st_size,
            "legacy_contract_sha256": legacy_contract,
            "optional_self_sha256": optional_self,
            "declared_source_runtime_contract_sha256": audit.get(
                "source_runtime_contract_sha256"
            ),
            "frozen_source_runtime_contract_sha256": frozen[
                "source_runtime_contract_sha256"
            ],
            "declared_validation_sha256": declared_validation,
        },
        "validation": {
            "actual_file_sha256": actual_validation,
            "actual_size_bytes": validation_path.stat().st_size,
            "frozen_expected_sha256": frozen["validation_sha256"],
            "frozen_expected_rows_not_read": frozen["validation_rows"],
        },
        "equality": equality,
        "classification": classify(equality),
    }


def diagnose(
    *,
    archive_path: Path,
    bundle_root: Path,
    bundle_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    manifest_self_sha256: str,
    revision: str,
    work_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or work_dir.exists():
        raise FileExistsError("refusing to overwrite fold0 diagnostic output")
    if not HEX64.fullmatch(bundle_sha256):
        raise ValueError("diagnostic bundle SHA must be exact")
    manifest = validate_bundle(
        bundle_root,
        manifest_path,
        expected_manifest_sha256=manifest_sha256,
        expected_manifest_self_sha256=manifest_self_sha256,
        expected_revision=revision,
    )
    members = {item["path"]: item for item in manifest["files"]}
    transport = _load_module(bundle_root / EXTRACTOR_PATH)
    spec, _ = transport.validate_source_prepare_spec(bundle_root / SPEC_PATH)
    extraction_root = work_dir / "extracted"
    transport_report_path = work_dir / "transport_report.json"
    work_dir.mkdir(parents=True, exist_ok=False)
    transport_report = transport.extract(
        archive_path=archive_path,
        destination=extraction_root,
        report_path=transport_report_path,
        archive_id="source_f03",
    )
    inspected = inspect_extracted(extraction_root, spec=spec)
    report = {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": "689",
        "scope": "fold0_runtime_audit_validation_binding_remote_cpu_diagnostic",
        "code": {
            "revision": revision,
            "bundle_sha256": bundle_sha256,
            "manifest_file_sha256": manifest_sha256,
            "manifest_self_sha256": manifest_self_sha256,
            "runner_sha256": members[RUNNER_PATH]["sha256"],
            "extractor_sha256": members[EXTRACTOR_PATH]["sha256"],
            "source_spec_file_sha256": members[SPEC_PATH]["sha256"],
            "source_spec_self_sha256": spec["self_sha256"],
        },
        "archive": {
            "archive_id": "source_f03",
            "sha256": transport_report["archive_sha256"],
            "size_bytes": transport_report["archive_size_bytes"],
            "member_count": transport_report["member_count"],
            "type_counts": transport_report["type_counts"],
            "apple_metadata_skipped_count": transport_report[
                "apple_metadata_skipped_count"
            ],
            "profile_match": True,
            "transport_report_file_sha256": sha256_file(transport_report_path),
            "transport_report_self_sha256": transport_report["self_sha256"],
        },
        **inspected,
        "execution_scope": "remote_cpu_only",
        "labels_read": 0,
        "rows_read": 0,
        "sealed_rows": 0,
        "public_rows": 0,
        "retry_authorized": False,
        "teacher_authorized": False,
        "student_authorized": False,
        "self_sha256": None,
    }
    report["self_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--archive", dest="archive_path", type=Path, required=True)
    value.add_argument("--bundle-root", type=Path, required=True)
    value.add_argument("--bundle-sha256", required=True)
    value.add_argument("--manifest", dest="manifest_path", type=Path, required=True)
    value.add_argument("--manifest-sha256", required=True)
    value.add_argument("--manifest-self-sha256", required=True)
    value.add_argument("--revision", required=True)
    value.add_argument("--work-dir", type=Path, required=True)
    value.add_argument("--output", dest="output_path", type=Path, required=True)
    return value


def main() -> None:
    print(json.dumps(diagnose(**vars(parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
