"""Independently verify the exp689 fold0 validation-binding diagnostic."""

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
RUNNER = f"{EXPERIMENT}/diagnose_fold0_validation_binding.py"
EXTRACTOR = f"{EXPERIMENT}/extract_source_archive_transport.py"
SPEC = f"{EXPERIMENT}/source_prepare_spec_v1.json"
BUNDLE_PATHS = {RUNNER, EXTRACTOR, SPEC}
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
REPORT_FIELDS = {
    "schema_version",
    "experiment_id",
    "scope",
    "code",
    "archive",
    "paths",
    "runtime_audit",
    "validation",
    "equality",
    "classification",
    "execution_scope",
    "labels_read",
    "rows_read",
    "sealed_rows",
    "public_rows",
    "retry_authorized",
    "teacher_authorized",
    "student_authorized",
    "self_sha256",
}


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


def _json(path: Path, context: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp689_fold0_verify_transport", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load accepted transport extractor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _self_hash(value: dict[str, Any]) -> str:
    actual = value.get("self_sha256")
    copy = dict(value)
    copy["self_sha256"] = None
    if (
        not isinstance(actual, str)
        or not HEX64.fullmatch(actual)
        or hashlib.sha256(canonical_json_bytes(copy)).hexdigest() != actual
    ):
        raise ValueError("diagnostic canonical self SHA mismatch")
    return actual


def _manifest(
    root: Path,
    path: Path,
    *,
    file_sha256: str,
    self_sha256: str,
    revision: str,
) -> dict[str, Any]:
    if (
        not HEX40.fullmatch(revision)
        or not HEX64.fullmatch(file_sha256)
        or not HEX64.fullmatch(self_sha256)
        or sha256_file(path) != file_sha256
    ):
        raise ValueError("diagnostic manifest external binding mismatch")
    value = _json(path, "diagnostic manifest")
    if set(value) != {"schema_version", "builder_revision", "files", "self_sha256"}:
        raise ValueError("diagnostic manifest exact schema mismatch")
    actual_self = value["self_sha256"]
    copy = dict(value)
    copy["self_sha256"] = None
    files = value["files"]
    if (
        actual_self != self_sha256
        or hashlib.sha256(canonical_json_bytes(copy)).hexdigest() != actual_self
        or value["schema_version"] != BUNDLE_SCHEMA
        or value["builder_revision"] != revision
        or not isinstance(files, list)
        or [item.get("path") for item in files] != sorted(BUNDLE_PATHS)
    ):
        raise ValueError("diagnostic manifest identity mismatch")
    actual_paths = {
        item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()
    }
    if actual_paths != BUNDLE_PATHS:
        raise ValueError("diagnostic bundle whitelist mismatch")
    for item in files:
        member = root / item["path"]
        if (
            set(item) != {"path", "sha256", "size_bytes"}
            or not member.is_file()
            or member.is_symlink()
            or member.stat().st_size != item["size_bytes"]
            or sha256_file(member) != item["sha256"]
        ):
            raise ValueError("diagnostic bundle member mismatch")
    return value


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


def _classification(equality: dict[str, bool | None]) -> str:
    if equality["runtime_audit_contract_valid"] is not True:
        return "RUNTIME_AUDIT_CONTRACT_INVALID"
    if equality["source_runtime_contract_equals_frozen"] is not True:
        return "SOURCE_RUNTIME_CONTRACT_MISMATCH"
    if all(
        equality[field] is True
        for field in (
            "declared_equals_actual",
            "declared_equals_frozen",
            "actual_equals_frozen",
        )
    ):
        return "ALL_VALIDATION_BINDINGS_EQUAL"
    if equality["declared_equals_frozen"] and not equality["actual_equals_frozen"]:
        return "ACTUAL_VALIDATION_DIFFERS_FROM_DECLARED_AND_FROZEN"
    if equality["actual_equals_frozen"] and not equality["declared_equals_frozen"]:
        return "DECLARED_VALIDATION_DIFFERS_FROM_ACTUAL_AND_FROZEN"
    if equality["declared_equals_actual"] and not equality["actual_equals_frozen"]:
        return "FROZEN_VALIDATION_EXPECTATION_DIFFERS"
    return "VALIDATION_BINDINGS_ALL_DIFFERENT_OR_MISSING"


def verify(
    *,
    report_path: Path,
    report_sha256: str,
    report_self_sha256: str,
    archive_path: Path,
    bundle_root: Path,
    bundle_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    manifest_self_sha256: str,
    revision: str,
    work_dir: Path,
) -> dict[str, Any]:
    for value in (
        report_sha256,
        report_self_sha256,
        bundle_sha256,
        manifest_sha256,
        manifest_self_sha256,
    ):
        if not HEX64.fullmatch(value):
            raise ValueError("independent verifier requires exact SHA-256 bindings")
    if work_dir.exists():
        raise FileExistsError("refusing to overwrite verifier work directory")
    manifest = _manifest(
        bundle_root,
        manifest_path,
        file_sha256=manifest_sha256,
        self_sha256=manifest_self_sha256,
        revision=revision,
    )
    if sha256_file(report_path) != report_sha256:
        raise ValueError("diagnostic report file SHA mismatch")
    report = _json(report_path, "diagnostic report")
    if set(report) != REPORT_FIELDS or _self_hash(report) != report_self_sha256:
        raise ValueError("diagnostic report exact schema/self mismatch")
    members = {item["path"]: item for item in manifest["files"]}
    transport = _module(bundle_root / EXTRACTOR)
    spec, _ = transport.validate_source_prepare_spec(bundle_root / SPEC)
    work_dir.mkdir(parents=True, exist_ok=False)
    extracted = work_dir / "extracted"
    transport_path = work_dir / "transport_report.json"
    transport_report = transport.extract(
        archive_path=archive_path,
        destination=extracted,
        report_path=transport_path,
        archive_id="source_f03",
    )
    audit_path = extracted / AUDIT_MEMBER
    validation_path = extracted / VALIDATION_MEMBER
    if any(
        not path.is_file() or path.is_symlink()
        for path in (audit_path, validation_path)
    ):
        raise ValueError("exact fold0 diagnostic path is missing or substituted")
    audit = _json(audit_path, "fold0 runtime_audit")
    frozen = spec["folds"]["0"]
    outputs = audit.get("output_sha256")
    declared = outputs.get("validation.jsonl") if isinstance(outputs, dict) else None
    actual = sha256_file(validation_path)
    contract = _canonical_field(audit, "contract_sha256", remove=True)
    optional_self = _canonical_field(audit, "self_sha256", remove=False)
    equality: dict[str, bool | None] = {
        "declared_equals_actual": declared == actual,
        "declared_equals_frozen": declared == frozen["validation_sha256"],
        "actual_equals_frozen": actual == frozen["validation_sha256"],
        "source_runtime_contract_equals_frozen": (
            audit.get("source_runtime_contract_sha256")
            == frozen["source_runtime_contract_sha256"]
        ),
        "runtime_audit_contract_valid": contract["valid"],
        "runtime_audit_self_valid": optional_self["valid"],
    }
    expected = {
        "code": {
            "revision": revision,
            "bundle_sha256": bundle_sha256,
            "manifest_file_sha256": manifest_sha256,
            "manifest_self_sha256": manifest_self_sha256,
            "runner_sha256": members[RUNNER]["sha256"],
            "extractor_sha256": members[EXTRACTOR]["sha256"],
            "source_spec_file_sha256": members[SPEC]["sha256"],
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
            "transport_report_file_sha256": sha256_file(transport_path),
            "transport_report_self_sha256": transport_report["self_sha256"],
        },
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
            "legacy_contract_sha256": contract,
            "optional_self_sha256": optional_self,
            "declared_source_runtime_contract_sha256": audit.get(
                "source_runtime_contract_sha256"
            ),
            "frozen_source_runtime_contract_sha256": frozen[
                "source_runtime_contract_sha256"
            ],
            "declared_validation_sha256": declared,
        },
        "validation": {
            "actual_file_sha256": actual,
            "actual_size_bytes": validation_path.stat().st_size,
            "frozen_expected_sha256": frozen["validation_sha256"],
            "frozen_expected_rows_not_read": frozen["validation_rows"],
        },
        "equality": equality,
        "classification": _classification(equality),
    }
    if any(report[field] != value for field, value in expected.items()):
        raise ValueError("diagnostic report differs from independent recomputation")
    if (
        report["schema_version"] != REPORT_SCHEMA
        or report["experiment_id"] != "689"
        or report["scope"]
        != "fold0_runtime_audit_validation_binding_remote_cpu_diagnostic"
        or report["execution_scope"] != "remote_cpu_only"
        or any(report[field] != 0 for field in ("labels_read", "rows_read", "sealed_rows", "public_rows"))
        or any(
            report[field] is not False
            for field in ("retry_authorized", "teacher_authorized", "student_authorized")
        )
    ):
        raise ValueError("diagnostic report scope/safety mismatch")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--report", dest="report_path", type=Path, required=True)
    value.add_argument("--report-sha256", required=True)
    value.add_argument("--report-self-sha256", required=True)
    value.add_argument("--archive", dest="archive_path", type=Path, required=True)
    value.add_argument("--bundle-root", type=Path, required=True)
    value.add_argument("--bundle-sha256", required=True)
    value.add_argument("--manifest", dest="manifest_path", type=Path, required=True)
    value.add_argument("--manifest-sha256", required=True)
    value.add_argument("--manifest-self-sha256", required=True)
    value.add_argument("--revision", required=True)
    value.add_argument("--work-dir", type=Path, required=True)
    return value


def main() -> None:
    print(json.dumps(verify(**vars(parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
