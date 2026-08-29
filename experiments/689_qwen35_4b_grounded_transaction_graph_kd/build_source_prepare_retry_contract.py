"""Build the non-launchable semantic contract for one exp689 PREPARE retry."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import build_source_prepare_retry_preset as preset
import extract_source_archive_transport as transport
from build_source_prepare_preset import ALLOWED_REGIONS, safe_key

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def validate_args(args: argparse.Namespace) -> None:
    if args.region not in ALLOWED_REGIONS:
        raise ValueError("retry contract region is not approved")
    if not HEX40.fullmatch(args.revision):
        raise ValueError("retry contract revision must be an exact Git SHA")
    for field in (
        "bundle_sha256",
        "manifest_sha256",
        "manifest_self_sha256",
        "source_f03_sha256",
        "source_f124_sha256",
        "exclusion_670_sha256",
        "exclusion_672_sha256",
        "archive_acceptance_sha256",
        "archive_acceptance_self_sha256",
        "archive_verifier_terminal_metadata_sha256",
        "retry_preset_builder_sha256",
    ):
        if not HEX64.fullmatch(getattr(args, field)):
            raise ValueError(f"{field} must be an exact lowercase SHA-256")
    for field in (
        "bundle_key",
        "manifest_key",
        "source_f03_key",
        "source_f124_key",
        "exclusion_670_key",
        "exclusion_672_key",
        "archive_acceptance_key",
        "retry_preset_contract_key",
        "transport_retry_gate_key",
        "output_prefix",
        "transport_report_prefix",
    ):
        safe_key(getattr(args, field))
    if (
        args.source_f03_sha256 != transport.PROFILES["source_f03"]["sha256"]
        or args.source_f124_sha256
        != transport.PROFILES["source_f124"]["sha256"]
        or args.exclusion_670_sha256
        != transport.EXCLUSION_BINDINGS["exp670_audit_csv_sha256"]
        or args.exclusion_672_sha256
        != transport.EXCLUSION_BINDINGS["exp672_private_manifest_sha256"]
    ):
        raise ValueError("retry contract frozen source/exclusion binding mismatch")


def build(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    return preset.preset_contract_document(args)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--region", required=True)
    value.add_argument("--bucket", required=True)
    value.add_argument("--revision", required=True)
    for name in (
        "bundle",
        "manifest",
        "source-f03",
        "source-f124",
        "exclusion-670",
        "exclusion-672",
        "archive-acceptance",
    ):
        value.add_argument(f"--{name}-key", required=True)
        value.add_argument(f"--{name}-sha256", required=True)
    value.add_argument("--manifest-self-sha256", required=True)
    value.add_argument("--archive-acceptance-self-sha256", required=True)
    value.add_argument(
        "--archive-verifier-terminal-metadata-sha256", required=True
    )
    value.add_argument("--retry-preset-builder-sha256", required=True)
    value.add_argument("--retry-preset-contract-key", required=True)
    value.add_argument("--transport-retry-gate-key", required=True)
    value.add_argument("--output-prefix", required=True)
    value.add_argument("--transport-report-prefix", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite retry contract")
    document = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
