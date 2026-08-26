"""Transport-aware post-terminal verifier for the exp689 PREPARE retry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
RUNNER_PATH = f"{EXPERIMENT}/verify_source_prepare_retry.py"
RETRY_PATHS = {
    f"{EXPERIMENT}/extract_source_archive_transport.py",
    f"{EXPERIMENT}/prepare_source_universe.py",
    f"{EXPERIMENT}/source_prepare_spec_v1.json",
    f"{EXPERIMENT}/verify_source_prepare.py",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
S3_REF = re.compile(r"^s3://[^/?#]+/[^?#]+$")
TRANSPORT_REPORT_FIELDS = {
    "schema_version",
    "archive_id",
    "archive_sha256",
    "archive_size_bytes",
    "member_count",
    "type_counts",
    "apple_metadata_skipped_count",
    "apple_metadata_skipped_names",
    "extracted_regular_files",
    "extracted_directories",
    "symlinks_extracted",
    "hardlinks_extracted",
    "path_traversal_extracted",
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _self_hash(value: dict[str, Any], context: str) -> str:
    actual = value.get("self_sha256")
    if not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise ValueError(f"{context} requires an exact self SHA")
    copy = dict(value)
    copy["self_sha256"] = None
    if sha256_bytes(canonical_json_bytes(copy)) != actual:
        raise ValueError(f"{context} canonical self SHA mismatch")
    return actual


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_code_manifest(
    *,
    root: Path,
    manifest_path: Path,
    expected_file_sha256: str,
    expected_self_sha256: str,
    expected_revision: str,
    expected_schema: str,
    expected_paths: set[str],
) -> dict[str, Any]:
    if not HEX40.fullmatch(expected_revision):
        raise ValueError("code manifest revision must be an exact Git SHA")
    if not HEX64.fullmatch(expected_file_sha256) or not HEX64.fullmatch(
        expected_self_sha256
    ):
        raise ValueError("code manifest requires exact SHA-256 bindings")
    if sha256_file(manifest_path) != expected_file_sha256:
        raise ValueError("code manifest file SHA mismatch")
    manifest = _json(manifest_path, "code manifest")
    if set(manifest) != {"schema_version", "builder_revision", "files", "self_sha256"}:
        raise ValueError("code manifest exact schema mismatch")
    if _self_hash(manifest, "code manifest") != expected_self_sha256:
        raise ValueError("code manifest expected self SHA mismatch")
    files = manifest["files"]
    if (
        manifest["schema_version"] != expected_schema
        or manifest["builder_revision"] != expected_revision
        or not isinstance(files, list)
        or [item.get("path") for item in files] != sorted(expected_paths)
    ):
        raise ValueError("code manifest frozen identity mismatch")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_paths:
        raise ValueError("code bundle exact file inventory mismatch")
    for index, item in enumerate(files):
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"code manifest file {index} schema mismatch")
        path = root / item["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"code manifest file {index} content mismatch")
    return manifest


def _expected_ref(contract: dict[str, Any], key: str) -> str:
    value = f"s3://{contract['bucket']}{key}"
    if not S3_REF.fullmatch(value):
        raise ValueError("retry contract contains an unsafe S3 reference")
    return value


def _validate_transport_report(
    path: Path, archive_id: str, profile: dict[str, Any]
) -> dict[str, Any]:
    report = _json(path, f"{archive_id} transport report")
    if set(report) != TRANSPORT_REPORT_FIELDS:
        raise ValueError(f"{archive_id} transport report exact schema mismatch")
    self_sha = _self_hash(report, f"{archive_id} transport report")
    skipped = report["apple_metadata_skipped_names"]
    if (
        report["schema_version"]
        != "exp689_source_archive_transport_extraction_v1"
        or report["archive_id"] != archive_id
        or report["archive_sha256"] != profile["sha256"]
        or report["archive_size_bytes"] != profile["size_bytes"]
        or report["member_count"] != profile["member_count"]
        or report["type_counts"] != profile["type_counts"]
        or report["apple_metadata_skipped_count"]
        != profile["apple_metadata_count"]
        or not isinstance(skipped, list)
        or len(skipped) != profile["apple_metadata_count"]
        or len(skipped) != len(set(skipped))
        or report["extracted_regular_files"]
        != profile["type_counts"].get("regular", 0)
        - profile["apple_metadata_count"]
        or report["extracted_directories"]
        != profile["type_counts"].get("directory", 0)
        or any(
            report[field] != 0
            for field in (
                "symlinks_extracted",
                "hardlinks_extracted",
                "path_traversal_extracted",
            )
        )
    ):
        raise ValueError(f"{archive_id} transport report frozen result mismatch")
    return {
        "archive_id": archive_id,
        "archive_sha256": report["archive_sha256"],
        "report_file_sha256": sha256_file(path),
        "report_self_sha256": self_sha,
        "report_size_bytes": path.stat().st_size,
        "apple_metadata_skipped_count": report["apple_metadata_skipped_count"],
        "apple_metadata_skipped_names_sha256": sha256_bytes(
            canonical_json_bytes(skipped)
        ),
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    hash_fields = (
        "verifier_bundle_sha256",
        "verifier_manifest_sha256",
        "verifier_manifest_self_sha256",
        "verifier_runner_sha256",
        "retry_bundle_sha256",
        "retry_manifest_sha256",
        "retry_manifest_self_sha256",
        "extractor_sha256",
        "source_f03_sha256",
        "source_f124_sha256",
        "exclusion_670_sha256",
        "exclusion_672_sha256",
        "diagnostic_acceptance_sha256",
        "diagnostic_acceptance_self_sha256",
        "diagnostic_verifier_terminal_metadata_sha256",
        "retry_contract_sha256",
        "retry_contract_self_sha256",
        "retry_gate_sha256",
        "retry_gate_self_sha256",
        "retry_preset_builder_sha256",
        "terminal_metadata_sha256",
    )
    if any(not HEX64.fullmatch(getattr(args, field)) for field in hash_fields):
        raise ValueError("transport-aware verifier requires exact SHA-256 bindings")
    if not HEX40.fullmatch(args.verifier_revision) or not HEX40.fullmatch(
        args.retry_revision
    ):
        raise ValueError("transport-aware verifier requires exact Git revisions")
    for path in (
        args.verifier_manifest,
        args.retry_manifest,
        args.source_f03,
        args.source_f124,
        args.exclusion_670,
        args.exclusion_672,
        args.diagnostic_acceptance,
        args.retry_contract,
        args.retry_gate,
        args.terminal_metadata,
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError("verifier inputs must be regular non-symlink files")
    for path in (
        args.base_acceptance,
        args.transport_report_dir,
        args.runtime_root,
        args.acceptance,
    ):
        if path.exists():
            raise FileExistsError("refusing to overwrite verifier output")

    verifier_manifest = _validate_code_manifest(
        root=args.verifier_bundle_root,
        manifest_path=args.verifier_manifest,
        expected_file_sha256=args.verifier_manifest_sha256,
        expected_self_sha256=args.verifier_manifest_self_sha256,
        expected_revision=args.verifier_revision,
        expected_schema="exp689_source_prepare_retry_verifier_bundle_manifest_v1",
        expected_paths={RUNNER_PATH},
    )
    runner = args.verifier_bundle_root / RUNNER_PATH
    if (
        sha256_file(runner) != args.verifier_runner_sha256
        or verifier_manifest["files"][0]["sha256"] != args.verifier_runner_sha256
    ):
        raise ValueError("transport-aware verifier runner SHA mismatch")
    retry_manifest = _validate_code_manifest(
        root=args.retry_bundle_root,
        manifest_path=args.retry_manifest,
        expected_file_sha256=args.retry_manifest_sha256,
        expected_self_sha256=args.retry_manifest_self_sha256,
        expected_revision=args.retry_revision,
        expected_schema="exp689_source_prepare_bundle_manifest_v2_transport",
        expected_paths=RETRY_PATHS,
    )
    extractor_path = args.retry_bundle_root / f"{EXPERIMENT}/extract_source_archive_transport.py"
    if (
        sha256_file(extractor_path) != args.extractor_sha256
        or next(
            item["sha256"]
            for item in retry_manifest["files"]
            if item["path"].endswith("extract_source_archive_transport.py")
        )
        != args.extractor_sha256
    ):
        raise ValueError("gate-authorized extractor SHA mismatch")

    transport = _load_module("exp689_retry_transport", extractor_path)
    source_verifier = _load_module(
        "exp689_retry_source_verifier",
        args.retry_bundle_root / f"{EXPERIMENT}/verify_source_prepare.py",
    )
    contract = transport.validate_retry_preset_contract(
        args.retry_contract,
        expected_file_sha256=args.retry_contract_sha256,
        expected_self_sha256=args.retry_contract_self_sha256,
    )
    diagnostic = transport.validate_diagnostic_acceptance(
        args.diagnostic_acceptance, args.diagnostic_acceptance_sha256
    )
    if diagnostic["self_sha256"] != args.diagnostic_acceptance_self_sha256:
        raise ValueError("diagnostic acceptance expected self SHA mismatch")
    transport.validate_transport_retry_gate(
        args.retry_gate,
        expected_file_sha256=args.retry_gate_sha256,
        expected_self_sha256=args.retry_gate_self_sha256,
        diagnostic_acceptance=diagnostic,
        diagnostic_acceptance_file_sha256=args.diagnostic_acceptance_sha256,
        expected_verifier_terminal_metadata_sha256=(
            args.diagnostic_verifier_terminal_metadata_sha256
        ),
        expected_retry_code_commit=args.retry_revision,
        expected_retry_code_bundle_sha256=args.retry_bundle_sha256,
        expected_retry_bundle_manifest_file_sha256=args.retry_manifest_sha256,
        expected_retry_bundle_manifest_self_sha256=args.retry_manifest_self_sha256,
        expected_retry_preset_builder_sha256=args.retry_preset_builder_sha256,
        retry_preset_contract=contract,
        retry_preset_contract_file_sha256=args.retry_contract_sha256,
        expected_retry_preset_contract_sha256=args.retry_contract_self_sha256,
        source_prepare_spec_path=(
            args.retry_bundle_root / f"{EXPERIMENT}/source_prepare_spec_v1.json"
        ),
        expected_output_prefix=args.prepare_output_prefix,
        expected_transport_report_prefix=args.prepare_transport_report_prefix,
    )

    expected_refs = {
        "retry_bundle_ref": _expected_ref(contract, contract["inputs"]["bundle"]["key"]),
        "retry_manifest_ref": _expected_ref(
            contract, contract["inputs"]["manifest"]["key"]
        ),
        "source_f03_ref": _expected_ref(
            contract, contract["inputs"]["source_f03"]["key"]
        ),
        "source_f124_ref": _expected_ref(
            contract, contract["inputs"]["source_f124"]["key"]
        ),
        "exclusion_670_ref": _expected_ref(
            contract, contract["inputs"]["exclusion_670"]["key"]
        ),
        "exclusion_672_ref": _expected_ref(
            contract, contract["inputs"]["exclusion_672"]["key"]
        ),
        "diagnostic_acceptance_ref": _expected_ref(
            contract, contract["inputs"]["diagnostic_acceptance"]["key"]
        ),
        "retry_contract_ref": _expected_ref(
            contract, contract["inputs"]["preset_contract_key"]
        ),
        "retry_gate_ref": _expected_ref(
            contract, contract["inputs"]["transport_retry_gate_key"]
        ),
    }
    if any(getattr(args, field) != expected for field, expected in expected_refs.items()):
        raise ValueError("S3 input reference differs from gate-authorized retry contract")
    approved_output_ref = _expected_ref(contract, contract["outputs"]["source_prepare"])
    if (
        args.prepare_output_ref != approved_output_ref
        or args.prepare_output_prefix != contract["outputs"]["source_prepare"]
        or args.prepare_transport_report_prefix
        != contract["outputs"]["transport_reports"]
    ):
        raise ValueError("PREPARE output identity differs from retry contract")
    if (
        args.source_f03_sha256 != transport.PROFILES["source_f03"]["sha256"]
        or args.source_f124_sha256 != transport.PROFILES["source_f124"]["sha256"]
        or args.exclusion_670_sha256
        != transport.EXCLUSION_BINDINGS["exp670_audit_csv_sha256"]
        or args.exclusion_672_sha256
        != transport.EXCLUSION_BINDINGS["exp672_private_manifest_sha256"]
    ):
        raise ValueError("frozen archive/exclusion SHA mismatch")

    args.transport_report_dir.mkdir(parents=True, exist_ok=False)
    args.runtime_root.mkdir(parents=True, exist_ok=False)
    transport_reports: dict[str, dict[str, Any]] = {}
    runtime_destinations = {
        "source_f03": args.runtime_root / "source_f03",
        "source_f124": args.runtime_root / "source_f124",
    }
    archive_paths = {
        "source_f03": args.source_f03,
        "source_f124": args.source_f124,
    }
    for archive_id in ("source_f03", "source_f124"):
        report_path = args.transport_report_dir / f"{archive_id}_extraction.json"
        transport.extract(
            archive_path=archive_paths[archive_id],
            destination=runtime_destinations[archive_id],
            report_path=report_path,
            archive_id=archive_id,
        )
        transport_reports[archive_id] = _validate_transport_report(
            report_path, archive_id, transport.PROFILES[archive_id]
        )

    args.base_acceptance.parent.mkdir(parents=True, exist_ok=True)
    base = source_verifier.verify(
        prepare_dir=args.prepare_dir,
        runtime_dirs=[
            runtime_destinations["source_f03"]
            / "experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold0",
            runtime_destinations["source_f124"] / "runtime/fold1",
            runtime_destinations["source_f124"] / "runtime/fold2",
            runtime_destinations["source_f03"]
            / "experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold3",
            runtime_destinations["source_f124"] / "runtime/fold4",
        ],
        runtime_archives=[args.source_f03, args.source_f124],
        runtime_archive_refs=[args.source_f03_ref, args.source_f124_ref],
        exclusion_670_path=args.exclusion_670,
        exclusion_672_path=args.exclusion_672,
        exclusion_670_sha256=args.exclusion_670_sha256,
        exclusion_672_sha256=args.exclusion_672_sha256,
        bundle_root=args.retry_bundle_root,
        bundle_manifest_path=args.retry_manifest,
        bundle_manifest_sha256=args.retry_manifest_sha256,
        builder_revision=args.retry_revision,
        spec_path=(args.retry_bundle_root / f"{EXPERIMENT}/source_prepare_spec_v1.json"),
        terminal_metadata_path=args.terminal_metadata,
        terminal_metadata_sha256=args.terminal_metadata_sha256,
        approved_s3_output_ref=args.prepare_output_ref,
        acceptance_path=args.base_acceptance,
    )
    expected_runtime_archives = [
        {
            "archive_id": "folds_0_3",
            "folds": [0, 3],
            "reference": args.source_f03_ref,
            "sha256": args.source_f03_sha256,
            "size_bytes": args.source_f03.stat().st_size,
        },
        {
            "archive_id": "folds_1_2_4",
            "folds": [1, 2, 4],
            "reference": args.source_f124_ref,
            "sha256": args.source_f124_sha256,
            "size_bytes": args.source_f124.stat().st_size,
        },
    ]
    if (
        base.get("schema_version") != "exp689_source_prepare_acceptance_v1"
        or base.get("decision") != "ACCEPT"
        or base.get("terminal_metadata_sha256") != args.terminal_metadata_sha256
        or base.get("approved_s3_output_ref") != args.prepare_output_ref
        or base.get("runtime_archives") != expected_runtime_archives
    ):
        raise ValueError("base source PREPARE acceptance identity mismatch")
    base_self = _self_hash(base, "base source PREPARE acceptance")
    if sha256_file(args.base_acceptance) != sha256_bytes(
        (json.dumps(base, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ):
        raise ValueError("base source PREPARE acceptance file serialization mismatch")
    terminal = _json(args.terminal_metadata, "resolved terminal metadata")
    terminal_self = _self_hash(terminal, "resolved terminal metadata")
    if terminal.get("status") != "SUCCESS" or terminal.get("output_ref") != args.prepare_output_ref:
        raise ValueError("resolved terminal metadata does not bind terminal PREPARE output")

    acceptance = {
        "schema_version": "exp689_source_prepare_retry_acceptance_v1",
        "decision": "ACCEPT_TRANSPORT_AWARE_RETRY",
        "verifier_revision": args.verifier_revision,
        "verifier_runner_sha256": args.verifier_runner_sha256,
        "verifier_bundle_sha256": args.verifier_bundle_sha256,
        "verifier_manifest_file_sha256": args.verifier_manifest_sha256,
        "verifier_manifest_self_sha256": args.verifier_manifest_self_sha256,
        "retry_revision": args.retry_revision,
        "retry_bundle_sha256": args.retry_bundle_sha256,
        "retry_manifest_file_sha256": args.retry_manifest_sha256,
        "retry_manifest_self_sha256": args.retry_manifest_self_sha256,
        "extractor_sha256": args.extractor_sha256,
        "diagnostic_acceptance_ref": args.diagnostic_acceptance_ref,
        "diagnostic_acceptance_file_sha256": args.diagnostic_acceptance_sha256,
        "diagnostic_acceptance_self_sha256": diagnostic["self_sha256"],
        "retry_contract_ref": args.retry_contract_ref,
        "retry_contract_file_sha256": args.retry_contract_sha256,
        "retry_contract_self_sha256": contract["self_sha256"],
        "retry_gate_ref": args.retry_gate_ref,
        "retry_gate_file_sha256": args.retry_gate_sha256,
        "retry_gate_self_sha256": args.retry_gate_self_sha256,
        "original_runtime_archives": base["runtime_archives"],
        "transport_reports": transport_reports,
        "source_prepare_acceptance_file_sha256": sha256_file(args.base_acceptance),
        "source_prepare_acceptance_self_sha256": base_self,
        "output_inventory_sha256": base["output_inventory_sha256"],
        "output_inventory": base["output_inventory"],
        "resolved_terminal_metadata_ref": args.terminal_metadata_ref,
        "resolved_terminal_metadata_file_sha256": args.terminal_metadata_sha256,
        "resolved_terminal_metadata_self_sha256": terminal_self,
        "terminal_job_id": base["terminal_job_id"],
        "terminal_status": base["terminal_status"],
        "terminal_finished_at": base["terminal_finished_at"],
        "approved_s3_output_ref": args.prepare_output_ref,
        "max_jobs": 1,
        "retry_attempt": 1,
        "teacher_authorized": False,
        "model_authorized": False,
        "review_authorized": False,
        "student_gpu_authorized": False,
        "public_used": False,
        "sealed_rows": 0,
        "public_rows": 0,
        "self_sha256": None,
    }
    acceptance["self_sha256"] = sha256_bytes(canonical_json_bytes(acceptance))
    args.acceptance.parent.mkdir(parents=True, exist_ok=True)
    args.acceptance.write_text(
        json.dumps(
            acceptance,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return acceptance


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("verifier", "retry"):
        value.add_argument(f"--{name}-bundle-root", type=Path, required=True)
        value.add_argument(f"--{name}-bundle-sha256", required=True)
        value.add_argument(f"--{name}-manifest", type=Path, required=True)
        value.add_argument(f"--{name}-manifest-sha256", required=True)
        value.add_argument(f"--{name}-manifest-self-sha256", required=True)
        value.add_argument(f"--{name}-revision", required=True)
    value.add_argument("--verifier-runner-sha256", required=True)
    value.add_argument("--extractor-sha256", required=True)
    for name in (
        "source-f03",
        "source-f124",
        "exclusion-670",
        "exclusion-672",
        "diagnostic-acceptance",
        "retry-contract",
        "retry-gate",
        "terminal-metadata",
    ):
        value.add_argument(f"--{name}", type=Path, required=True)
        value.add_argument(f"--{name}-sha256", required=True)
        value.add_argument(f"--{name}-ref", required=True)
    value.add_argument("--diagnostic-acceptance-self-sha256", required=True)
    value.add_argument(
        "--diagnostic-verifier-terminal-metadata-sha256", required=True
    )
    value.add_argument("--retry-contract-self-sha256", required=True)
    value.add_argument("--retry-gate-self-sha256", required=True)
    value.add_argument("--retry-preset-builder-sha256", required=True)
    value.add_argument("--retry-bundle-ref", required=True)
    value.add_argument("--retry-manifest-ref", required=True)
    value.add_argument("--prepare-output-prefix", required=True)
    value.add_argument("--prepare-transport-report-prefix", required=True)
    value.add_argument("--prepare-output-ref", required=True)
    value.add_argument("--prepare-dir", type=Path, required=True)
    value.add_argument("--runtime-root", type=Path, required=True)
    value.add_argument("--transport-report-dir", type=Path, required=True)
    value.add_argument("--base-acceptance", type=Path, required=True)
    value.add_argument("--acceptance", type=Path, required=True)
    return value


def main() -> None:
    result = verify(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
