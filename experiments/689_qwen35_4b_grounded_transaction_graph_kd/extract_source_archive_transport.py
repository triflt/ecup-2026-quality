"""Safely extract frozen exp689 source archives after the accepted header audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "source_f03": {
        "sha256": "e371c03a3fc893d990d38874e07200a5ac136b8c72f43109aa7f567761943ccd",
        "size_bytes": 15_852_324,
        "member_count": 138,
        "type_counts": {"directory": 21, "regular": 117},
        "apple_metadata_count": 69,
    },
    "source_f124": {
        "sha256": "1dfb9bf01a9567286051ee76a79fc4b41c2af360e5761b7a4663090745c5474d",
        "size_bytes": 15_000_958,
        "member_count": 10,
        "type_counts": {"regular": 10},
        "apple_metadata_count": 0,
    },
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ACCEPTANCE_FIELDS = {
    "schema_version",
    "status",
    "decision",
    "independent_remote_provenance_verified",
    "report_sha256",
    "report_size_bytes",
    "report_self_sha256",
    "diagnostic_code_commit",
    "diagnostic_code_sha256",
    "diagnostic_preset_sha256",
    "diagnostic_command_sha256",
    "submit_receipt_sha256",
    "resolved_metadata_sha256",
    "resolved_metadata_self_sha256",
    "verifier_commit",
    "verifier_sha256",
    "archive_bindings",
    "unsafe_reason_set",
    "prepare_retry_authorized",
    "teacher_authorized",
    "student_gpu_authorized",
    "self_sha256",
}
TRANSPORT_RETRY_GATE_FIELDS = {
    "schema_version",
    "experiment_id",
    "scope",
    "issuer_role",
    "decision",
    "diagnostic_acceptance_file_sha256",
    "diagnostic_acceptance_self_sha256",
    "diagnostic_verifier_terminal_metadata_sha256",
    "frozen_archives",
    "retry_code_commit",
    "retry_code_bundle_sha256",
    "retry_bundle_manifest_file_sha256",
    "retry_bundle_manifest_self_sha256",
    "retry_preset_builder_sha256",
    "retry_preset_contract_file_sha256",
    "retry_preset_contract_sha256",
    "source_prepare_spec_file_sha256",
    "source_prepare_spec_self_sha256",
    "selector_contract_sha256",
    "exclusion_bindings",
    "runtime_bindings",
    "output_prefix",
    "transport_report_prefix",
    "controlled_prepare_retry_authorized",
    "max_jobs",
    "teacher_authorized",
    "model_authorized",
    "review_authorized",
    "student_gpu_authorized",
    "public_used",
    "self_sha256",
}
SOURCE_PREPARE_SPEC_FILE_SHA256 = (
    "804e3b9802d8a9d2e766f32151d8398e4e20df8a998c5aee455ca5456f262ac0"
)
SOURCE_PREPARE_SPEC_SELF_SHA256 = (
    "c41f1d11e956af0262688810c1b1770a201feb49515b12b0239232dab3c65cf3"
)
EXCLUSION_BINDINGS = {
    "exp670_audit_csv_sha256": (
        "012def05a7370608cb959aa9a0d73326bc48b1f6d93cb1c9faea2bb7a8842b7a"
    ),
    "exp672_private_manifest_sha256": (
        "303b46c24af4aa781792884919d57bfcfb813386eb71056b91544d67bb0f28fd"
    ),
}
RUNTIME_BINDINGS = {
    "0": {
        "source_runtime_contract_sha256": "38802115365cef7e3a0c1a82abc5efc5ce92a41e0046f1ddad4ab9e02647c568",
        "validation_sha256": "109807781797f51ec8c1d98d0385aef2e9edd0d89183c4a44aab76e2a26f56bc",
        "validation_rows": 943,
    },
    "1": {
        "source_runtime_contract_sha256": "3d62eed9817bbbdb0ff4d55dd511904b4fa1dfef5db44deddb104dda0c60f576",
        "validation_sha256": "f6111b8977957e93469c033980853512dc865bfeebc6a9256b7fb082338895cc",
        "validation_rows": 943,
    },
    "2": {
        "source_runtime_contract_sha256": "321b5e5165110fc729598956d121208ab14aee12af38e8f0b3ab81ddbd52f5e8",
        "validation_sha256": "47b59ecd0a0eb50136052f24883ba07a2af75ddae5eb989fcf2a8dc1bf889a44",
        "validation_rows": 944,
    },
    "3": {
        "source_runtime_contract_sha256": "e08a51c2db16163953c45841f3dd1e7b30b293a7265b2bbd8084d1479c20ea36",
        "validation_sha256": "bbbbae3f4ddb38bf3af238cd53dd8fc1b851b04598101ab7b57809e73c7a2257",
        "validation_rows": 943,
    },
    "4": {
        "source_runtime_contract_sha256": "22cad9c7a1510a73b6ec606826839328329a9c0469f2fb00c45fd24ed9dcf33f",
        "validation_sha256": "1d1e089269041217aa7f197ce6a79ca6fcd849becc124780fe6511b92bdeca0c",
        "validation_rows": 943,
    },
}
SELECTOR_SPEC_FIELDS = (
    "category",
    "execution_scope",
    "exclusions",
    "folds",
    "forbidden_field_tokens",
    "image_transform",
    "input_data_sha256",
    "input_registry_sha256",
    "opaque_token_scheme",
    "ocr_policy",
    "quota_per_fold_per_stratum",
    "runtime_archives",
    "singleton_cue_free_per_fold",
    "strata",
    "validation_fields",
)
PRESET_CONTRACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "scope",
    "retry_attempt",
    "job",
    "bucket",
    "revision",
    "inputs",
    "outputs",
    "command_plan",
    "max_jobs",
    "teacher_authorized",
    "model_authorized",
    "review_authorized",
    "student_gpu_authorized",
    "public_used",
    "self_sha256",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_prepare_spec(path: Path) -> tuple[dict[str, Any], str]:
    if sha256_file(path) != SOURCE_PREPARE_SPEC_FILE_SHA256:
        raise ValueError("source PREPARE spec file SHA mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("source PREPARE spec must be an object")
    copy = dict(value)
    actual_self = copy.get("self_sha256")
    copy["self_sha256"] = None
    if (
        actual_self != SOURCE_PREPARE_SPEC_SELF_SHA256
        or hashlib.sha256(canonical_json_bytes(copy)).hexdigest() != actual_self
    ):
        raise ValueError("source PREPARE spec self-hash mismatch")
    if value.get("schema_version") != "exp689_source_prepare_spec_v1":
        raise ValueError("source PREPARE spec schema mismatch")
    selector = {field: value[field] for field in SELECTOR_SPEC_FIELDS}
    return value, hashlib.sha256(canonical_json_bytes(selector)).hexdigest()


def validate_retry_preset_contract(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_self_sha256: str,
) -> dict[str, Any]:
    if not HEX64.fullmatch(expected_file_sha256) or not HEX64.fullmatch(
        expected_self_sha256
    ):
        raise ValueError("retry preset contract requires exact hashes")
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("retry preset contract file SHA mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != PRESET_CONTRACT_FIELDS:
        raise ValueError("retry preset contract exact schema mismatch")
    copy = dict(value)
    actual_self = copy["self_sha256"]
    copy["self_sha256"] = None
    if (
        actual_self != expected_self_sha256
        or hashlib.sha256(canonical_json_bytes(copy)).hexdigest() != actual_self
    ):
        raise ValueError("retry preset contract self-hash mismatch")
    if (
        value["schema_version"]
        != "exp689_source_prepare_retry_preset_contract_v1"
        or value["experiment_id"] != "689"
        or value["scope"] != "source_prepare_transport_retry_only"
        or value["retry_attempt"] != 1
        or value["job"].get("gpu_count") != 0
        or value["max_jobs"] != 1
        or value["teacher_authorized"] is not False
        or value["model_authorized"] is not False
        or value["review_authorized"] is not False
        or value["student_gpu_authorized"] is not False
        or value["public_used"] is not False
    ):
        raise ValueError("retry preset contract authorization mismatch")
    expected_commands = [
        "safe_extract_exact_retry_bundle",
        "validate_exact_bundle_manifest",
        "validate_diagnostic_acceptance_false",
        "validate_independent_transport_retry_gate_true",
        "extract_frozen_f03_skipping_exact_apple_metadata",
        "extract_frozen_f124_skipping_zero_apple_metadata",
        "run_unchanged_prepare_source_universe",
    ]
    job = value["job"]
    inputs = value["inputs"]
    outputs = value["outputs"]
    if (
        set(job)
        != {
            "flavor",
            "time_limit",
            "region",
            "image",
            "preemption",
            "gpu_count",
        }
        or job["flavor"] != "8cpu-128ram"
        or job["time_limit"] != "2h"
        or job["region"] not in {"ix-m5-sm11", "ix-m5-sm12"}
        or job["image"] != "odsai/ecup26-quality-baseline:1.0"
        or job["preemption"] != "forbidden"
        or set(inputs)
        != {
            "bundle",
            "manifest",
            "source_f03",
            "source_f124",
            "exclusion_670",
            "exclusion_672",
            "diagnostic_acceptance",
            "diagnostic_verifier_terminal_metadata_sha256",
            "preset_builder_sha256",
            "preset_contract_key",
            "transport_retry_gate_key",
        }
        or set(outputs) != {"source_prepare", "transport_reports"}
        or value["command_plan"] != expected_commands
        or not isinstance(value["bucket"], str)
        or not value["bucket"]
        or not re.fullmatch(r"[0-9a-f]{40}", value["revision"])
    ):
        raise ValueError("retry preset contract semantic schema mismatch")
    for name in (
        "bundle",
        "source_f03",
        "source_f124",
        "exclusion_670",
        "exclusion_672",
    ):
        if set(inputs[name]) != {"key", "sha256"}:
            raise ValueError("retry preset contract input schema mismatch")
    if set(inputs["manifest"]) != {"key", "sha256", "self_sha256"} or set(
        inputs["diagnostic_acceptance"]
    ) != {"key", "sha256", "self_sha256"}:
        raise ValueError("retry preset contract self-hashed input schema mismatch")
    path_values = [
        *(item["key"] for item in inputs.values() if isinstance(item, dict)),
        inputs["preset_contract_key"],
        inputs["transport_retry_gate_key"],
        *outputs.values(),
    ]
    for raw in path_values:
        if not isinstance(raw, str):
            raise TypeError("retry preset contract S3 key must be a string")
        parsed = PurePosixPath(raw)
        if (
            not parsed.is_absolute()
            or ".." in parsed.parts
            or any(character.isspace() for character in raw)
            or any(character in raw for character in "?#\\")
        ):
            raise ValueError("retry preset contract contains an unsafe S3 key")
    return value


def validate_diagnostic_acceptance(
    path: Path, expected_file_sha256: str
) -> dict[str, Any]:
    if not HEX64.fullmatch(expected_file_sha256):
        raise ValueError("diagnostic acceptance requires an exact file SHA")
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("diagnostic acceptance file SHA mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != ACCEPTANCE_FIELDS:
        raise ValueError("diagnostic acceptance exact schema mismatch")
    copy = dict(value)
    actual_self = copy["self_sha256"]
    copy["self_sha256"] = None
    if (
        not isinstance(actual_self, str)
        or not HEX64.fullmatch(actual_self)
        or hashlib.sha256(canonical_json_bytes(copy)).hexdigest() != actual_self
    ):
        raise ValueError("diagnostic acceptance self-hash mismatch")
    expected_bindings = {
        archive_id: {
            "sha256": profile["sha256"],
            "size_bytes": profile["size_bytes"],
        }
        for archive_id, profile in PROFILES.items()
    }
    if (
        value["schema_version"] != "exp689_source_archive_audit_acceptance_v2"
        or value["status"] != "accepted_diagnostic_only"
        or value["decision"] != "TRANSPORT_CAUSE_APPLE_METADATA_ONLY"
        or value["independent_remote_provenance_verified"] is not True
        or value["report_sha256"]
        != "48e58fe07447544d8ebe1b3013ee6d003914864837b5a87a4c24ba5e6625cc67"
        or value["report_size_bytes"] != 21_769
        or value["diagnostic_code_commit"]
        != "076c009eaee106caba428c699627f864e302d292"
        or value["diagnostic_code_sha256"]
        != "20a55c6e1268881c2281cba8d53efad26b5ad58fdc527d2bbddaae7f5477fc75"
        or value["diagnostic_preset_sha256"]
        != "622370e98f66d1378a5e299d9d10401fd01b0d31f26e5191f3bea36321e59ddb"
        or value["diagnostic_command_sha256"]
        != "5e36767eb7b7744347bf643f14603ffdfc12344bcd09652dbf1ac12b05cf494b"
        or value["submit_receipt_sha256"]
        != "620c88baac32ac5edd8f8e630d861eda91c3f95a7286c50b3bcb5afba047436b"
        or value["resolved_metadata_sha256"]
        != "75eb921999b1256dfa3192617b38f7394a09274f165698d5287fde8f97722140"
        or value["resolved_metadata_self_sha256"]
        != "ef162d3abaf62637611d60df02c027851608138a32a3edb803f4d55768a188f0"
        or value["verifier_commit"]
        != "14d215ed4890903be10712dc18e717fc515d1bac"
        or value["verifier_sha256"]
        != "a8b125dfd232c3cb13afb18ceb6f3789e15958cb0d0b189fdffadee97590b41b"
        or value["archive_bindings"] != expected_bindings
        or value["unsafe_reason_set"] != ["apple_metadata"]
        or value["prepare_retry_authorized"] is not False
        or value["teacher_authorized"] is not False
        or value["student_gpu_authorized"] is not False
        or not isinstance(value["report_self_sha256"], str)
        or not HEX64.fullmatch(value["report_self_sha256"])
    ):
        raise ValueError("diagnostic acceptance frozen lineage/decision mismatch")
    return value


def validate_transport_retry_gate(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_self_sha256: str,
    diagnostic_acceptance: dict[str, Any],
    diagnostic_acceptance_file_sha256: str,
    expected_verifier_terminal_metadata_sha256: str,
    expected_retry_code_commit: str,
    expected_retry_code_bundle_sha256: str,
    expected_retry_bundle_manifest_file_sha256: str,
    expected_retry_bundle_manifest_self_sha256: str,
    expected_retry_preset_builder_sha256: str,
    retry_preset_contract: dict[str, Any],
    retry_preset_contract_file_sha256: str,
    expected_retry_preset_contract_sha256: str,
    source_prepare_spec_path: Path,
    expected_output_prefix: str,
    expected_transport_report_prefix: str,
) -> dict[str, Any]:
    """Require the independent authorization that the diagnostic cannot issue."""

    exact_hex = (
        expected_file_sha256,
        expected_self_sha256,
        diagnostic_acceptance_file_sha256,
        expected_verifier_terminal_metadata_sha256,
        expected_retry_code_bundle_sha256,
        expected_retry_bundle_manifest_file_sha256,
        expected_retry_bundle_manifest_self_sha256,
        expected_retry_preset_builder_sha256,
        retry_preset_contract_file_sha256,
        expected_retry_preset_contract_sha256,
    )
    if any(not HEX64.fullmatch(value) for value in exact_hex):
        raise ValueError("transport retry gate requires exact SHA-256 bindings")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_retry_code_commit):
        raise ValueError("transport retry gate requires an exact code commit")
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("transport retry gate file SHA mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != TRANSPORT_RETRY_GATE_FIELDS:
        raise ValueError("transport retry gate exact schema mismatch")
    copy = dict(value)
    actual_self = copy["self_sha256"]
    copy["self_sha256"] = None
    if (
        actual_self != expected_self_sha256
        or hashlib.sha256(canonical_json_bytes(copy)).hexdigest() != actual_self
    ):
        raise ValueError("transport retry gate self-hash mismatch")
    expected_archives = {
        archive_id: {
            "sha256": profile["sha256"],
            "size_bytes": profile["size_bytes"],
        }
        for archive_id, profile in PROFILES.items()
    }
    _, selector_contract_sha256 = validate_source_prepare_spec(
        source_prepare_spec_path
    )
    contract_inputs = retry_preset_contract["inputs"]
    if (
        retry_preset_contract["revision"] != expected_retry_code_commit
        or contract_inputs["bundle"]["sha256"]
        != expected_retry_code_bundle_sha256
        or contract_inputs["manifest"]["sha256"]
        != expected_retry_bundle_manifest_file_sha256
        or contract_inputs["manifest"]["self_sha256"]
        != expected_retry_bundle_manifest_self_sha256
        or contract_inputs["source_f03"]["sha256"]
        != PROFILES["source_f03"]["sha256"]
        or contract_inputs["source_f124"]["sha256"]
        != PROFILES["source_f124"]["sha256"]
        or contract_inputs["exclusion_670"]["sha256"]
        != EXCLUSION_BINDINGS["exp670_audit_csv_sha256"]
        or contract_inputs["exclusion_672"]["sha256"]
        != EXCLUSION_BINDINGS["exp672_private_manifest_sha256"]
        or contract_inputs["diagnostic_acceptance"]["sha256"]
        != diagnostic_acceptance_file_sha256
        or contract_inputs["diagnostic_acceptance"]["self_sha256"]
        != diagnostic_acceptance["self_sha256"]
        or contract_inputs["diagnostic_verifier_terminal_metadata_sha256"]
        != expected_verifier_terminal_metadata_sha256
        or contract_inputs["preset_builder_sha256"]
        != expected_retry_preset_builder_sha256
        or retry_preset_contract["outputs"]["source_prepare"]
        != expected_output_prefix
        or retry_preset_contract["outputs"]["transport_reports"]
        != expected_transport_report_prefix
    ):
        raise ValueError("retry preset contract frozen input/output mismatch")
    if (
        diagnostic_acceptance["prepare_retry_authorized"] is not False
        or value["schema_version"]
        != "exp689_source_prepare_transport_retry_gate_v1"
        or value["experiment_id"] != "689"
        or value["scope"] != "source_prepare_transport_retry_only"
        or value["issuer_role"] != "independent_integrator"
        or value["decision"] != "OPEN_CONTROLLED_SOURCE_PREPARE_RETRY"
        or value["diagnostic_acceptance_file_sha256"]
        != diagnostic_acceptance_file_sha256
        or value["diagnostic_acceptance_self_sha256"]
        != diagnostic_acceptance["self_sha256"]
        or value["diagnostic_verifier_terminal_metadata_sha256"]
        != expected_verifier_terminal_metadata_sha256
        or value["frozen_archives"] != expected_archives
        or value["retry_code_commit"] != expected_retry_code_commit
        or value["retry_code_bundle_sha256"]
        != expected_retry_code_bundle_sha256
        or value["retry_bundle_manifest_file_sha256"]
        != expected_retry_bundle_manifest_file_sha256
        or value["retry_bundle_manifest_self_sha256"]
        != expected_retry_bundle_manifest_self_sha256
        or value["retry_preset_builder_sha256"]
        != expected_retry_preset_builder_sha256
        or value["retry_preset_contract_file_sha256"]
        != retry_preset_contract_file_sha256
        or value["retry_preset_contract_sha256"]
        != expected_retry_preset_contract_sha256
        or retry_preset_contract["self_sha256"]
        != expected_retry_preset_contract_sha256
        or value["source_prepare_spec_file_sha256"]
        != SOURCE_PREPARE_SPEC_FILE_SHA256
        or value["source_prepare_spec_self_sha256"]
        != SOURCE_PREPARE_SPEC_SELF_SHA256
        or value["selector_contract_sha256"] != selector_contract_sha256
        or value["exclusion_bindings"] != EXCLUSION_BINDINGS
        or value["runtime_bindings"] != RUNTIME_BINDINGS
        or value["output_prefix"] != expected_output_prefix
        or value["transport_report_prefix"]
        != expected_transport_report_prefix
        or value["controlled_prepare_retry_authorized"] is not True
        or value["max_jobs"] != 1
        or value["teacher_authorized"] is not False
        or value["model_authorized"] is not False
        or value["review_authorized"] is not False
        or value["student_gpu_authorized"] is not False
        or value["public_used"] is not False
    ):
        raise ValueError("transport retry gate frozen authorization mismatch")
    return value


def kind(member: tarfile.TarInfo) -> str:
    if member.isfile():
        return "regular"
    if member.isdir():
        return "directory"
    return "forbidden"


def is_apple_metadata(path: PurePosixPath) -> bool:
    return any(
        part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
        for part in path.parts
    )


def validate_headers(
    members: list[tarfile.TarInfo], profile: dict[str, Any]
) -> tuple[list[tuple[tarfile.TarInfo, str]], list[str]]:
    normalized = [
        PurePosixPath(member.name).as_posix().removeprefix("./")
        for member in members
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate normalized tar member")
    if len(members) != profile["member_count"]:
        raise ValueError("frozen tar member count mismatch")
    type_counts = Counter(kind(member) for member in members)
    if dict(sorted(type_counts.items())) != profile["type_counts"]:
        raise ValueError("frozen tar member-type profile mismatch")
    safe: list[tuple[tarfile.TarInfo, str]] = []
    skipped: list[str] = []
    for member, name in zip(members, normalized, strict=True):
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not name:
            raise ValueError("unsafe tar path")
        member_kind = kind(member)
        apple = is_apple_metadata(path)
        if apple:
            if member_kind != "regular":
                raise ValueError("AppleDouble member is not a regular file")
            skipped.append(name)
            continue
        if member_kind not in {"regular", "directory"}:
            raise ValueError("non-regular tar member is forbidden")
        safe.append((member, name))
    if len(skipped) != profile["apple_metadata_count"]:
        raise ValueError("frozen AppleDouble member count mismatch")
    return safe, skipped


def extract(
    *,
    archive_path: Path,
    destination: Path,
    report_path: Path,
    archive_id: str,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if destination.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite extraction output")
    frozen = PROFILES[archive_id] if profile is None else profile
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError("source archive must be a regular file")
    if archive_path.stat().st_size != frozen["size_bytes"]:
        raise ValueError("frozen source archive size mismatch")
    actual_sha = sha256_file(archive_path)
    if actual_sha != frozen["sha256"]:
        raise ValueError("frozen source archive SHA mismatch")
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        safe, skipped = validate_headers(members, frozen)
        destination.mkdir(parents=True, exist_ok=False)
        extracted_files = 0
        extracted_directories = 0
        for member, name in safe:
            target = destination.joinpath(*PurePosixPath(name).parts)
            if member.isdir():
                if target.exists() and not target.is_dir():
                    raise ValueError("directory collides with extracted file")
                target.mkdir(parents=True, exist_ok=True)
                extracted_directories += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("regular member collides with existing path")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("regular member payload is unavailable")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1 << 20)
            extracted_files += 1
    report = {
        "schema_version": "exp689_source_archive_transport_extraction_v1",
        "archive_id": archive_id,
        "archive_sha256": actual_sha,
        "archive_size_bytes": archive_path.stat().st_size,
        "member_count": frozen["member_count"],
        "type_counts": frozen["type_counts"],
        "apple_metadata_skipped_count": len(skipped),
        "apple_metadata_skipped_names": skipped,
        "extracted_regular_files": extracted_files,
        "extracted_directories": extracted_directories,
        "symlinks_extracted": 0,
        "hardlinks_extracted": 0,
        "path_traversal_extracted": 0,
        "self_sha256": None,
    }
    report["self_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-id", choices=sorted(PROFILES), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = extract(
        archive_path=args.archive,
        destination=args.destination,
        report_path=args.report,
        archive_id=args.archive_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
