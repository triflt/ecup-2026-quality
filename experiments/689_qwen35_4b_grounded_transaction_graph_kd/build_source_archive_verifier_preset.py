"""Build a secret-free CPU preset for exp689 archive-diagnostic verification."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_REGION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@-]{0,255}$")
SAFE_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,62}$")
SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,1023}$")


@dataclass(frozen=True)
class S3Object:
    uri: str
    bucket: str
    key: str
    src: str
    file: str


def require_hex64(value: str, context: str) -> None:
    if not HEX64.fullmatch(value):
        raise ValueError(f"{context} must be an exact lowercase SHA-256")


def parse_s3_object(uri: str, context: str) -> S3Object:
    if "\\" in uri or "?" in uri or "#" in uri:
        raise ValueError(f"{context} contains forbidden URI syntax")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        raise ValueError(f"{context} must be an exact S3 object URI")
    key = parsed.path.removeprefix("/")
    pure = PurePosixPath(key)
    if (
        not SAFE_BUCKET.fullmatch(parsed.netloc)
        or not SAFE_KEY.fullmatch(key)
        or pure.is_absolute()
        or ".." in pure.parts
        or not pure.name
    ):
        raise ValueError(f"{context} contains an unsafe S3 key")
    src = "/" + pure.parent.as_posix() if pure.parent.as_posix() != "." else "/"
    return S3Object(
        uri=uri,
        bucket=parsed.netloc,
        key=key,
        src=src,
        file=pure.name,
    )


def parse_s3_prefix(uri: str, context: str) -> tuple[str, str]:
    if "\\" in uri or "?" in uri or "#" in uri:
        raise ValueError(f"{context} contains forbidden URI syntax")
    parsed = urlsplit(uri.rstrip("/"))
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        raise ValueError(f"{context} must be an exact S3 prefix URI")
    key = parsed.path.removeprefix("/")
    pure = PurePosixPath(key)
    if (
        not SAFE_BUCKET.fullmatch(parsed.netloc)
        or not SAFE_KEY.fullmatch(key)
        or pure.is_absolute()
        or ".." in pure.parts
    ):
        raise ValueError(f"{context} contains an unsafe S3 prefix")
    return parsed.netloc, "/" + pure.as_posix()


def input_lines(name: str, item: S3Object, dst: str) -> list[str]:
    return [
        "    - type: s3msk",
        f"      name: {json.dumps(name)}",
        f"      src: {json.dumps(item.src)}",
        f"      file: {json.dumps(item.file)}",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(item.bucket)}",
    ]


def build(args: argparse.Namespace) -> str:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite archive-verifier preset")
    if not HEX40.fullmatch(args.verifier_commit):
        raise ValueError("verifier commit must be an exact lowercase Git SHA")
    for value, context in (
        (args.verifier_sha256, "verifier SHA"),
        (args.report_sha256, "report SHA"),
        (args.receipt_sha256, "receipt SHA"),
        (args.metadata_sha256, "resolved metadata SHA"),
        (args.diagnostic_command_sha256, "diagnostic command SHA"),
    ):
        require_hex64(value, context)
    if args.report_size_bytes <= 0:
        raise ValueError("report size must be positive")
    if not SAFE_REGION.fullmatch(args.region):
        raise ValueError("region contains unsafe characters")
    if not SAFE_IMAGE.fullmatch(args.image):
        raise ValueError("image contains unsafe characters")
    verifier = parse_s3_object(args.verifier_uri, "verifier URI")
    report = parse_s3_object(args.report_uri, "report URI")
    receipt = parse_s3_object(args.receipt_uri, "receipt URI")
    metadata = parse_s3_object(args.metadata_uri, "resolved metadata URI")
    code = parse_s3_object(args.diagnostic_code_uri, "diagnostic code URI")
    source_f03 = parse_s3_object(args.source_f03_uri, "f03 source URI")
    source_f124 = parse_s3_object(args.source_f124_uri, "f124 source URI")
    diagnostic_bucket, diagnostic_dst = parse_s3_prefix(
        args.diagnostic_output_prefix, "diagnostic output prefix"
    )
    diagnostic_output_prefix = f"s3://{diagnostic_bucket}{diagnostic_dst}"
    output_bucket, output_dst = parse_s3_prefix(args.output_prefix, "output prefix")
    command = " ".join(
        [
            "test",
            '"$(sha256sum /work/input/verifier/verify_source_archive_audit.py | cut -d\' \' -f1)"',
            "=",
            f'"{args.verifier_sha256}"',
            "&&",
            "python3 -u /work/input/verifier/verify_source_archive_audit.py",
            "--report /work/input/report/archive_header_audit.json",
            f"--expected-report-sha256 {args.report_sha256}",
            f"--expected-report-size-bytes {args.report_size_bytes}",
            "--submit-receipt /work/input/receipt/submit_receipt.json",
            f"--expected-submit-receipt-sha256 {args.receipt_sha256}",
            "--resolved-metadata /work/input/metadata/resolved_metadata.json",
            f"--expected-resolved-metadata-sha256 {args.metadata_sha256}",
            f"--expected-command-sha256 {args.diagnostic_command_sha256}",
            f"--expected-code-uri {code.uri}",
            f"--expected-f03-uri {source_f03.uri}",
            f"--expected-f124-uri {source_f124.uri}",
            f"--approved-output-prefix {diagnostic_output_prefix}",
            f"--expected-verifier-sha256 {args.verifier_sha256}",
            f"--verifier-commit {args.verifier_commit}",
            "--output /work/output/source_archive_audit_acceptance.json",
        ]
    )
    lines = [
        "job:",
        "  generate_name: exp689-archive-verify",
        "  time_limit: 20m",
        "  flavor: 8cpu-128ram",
        f"  region: {args.region}",
        f"  image: {args.image}",
        "  preemption: forbidden",
        "  work_dir: /work",
        "  entrypoint: /bin/bash",
        "  args:",
        "    - -lc",
        "    - >-",
        f"      {command}",
        "  input:",
    ]
    for name, item, dst in (
        ("verifier", verifier, "/work/input/verifier/verify_source_archive_audit.py"),
        ("report", report, "/work/input/report/archive_header_audit.json"),
        ("receipt", receipt, "/work/input/receipt/submit_receipt.json"),
        ("metadata", metadata, "/work/input/metadata/resolved_metadata.json"),
    ):
        lines.extend(input_lines(name, item, dst))
    lines.extend(
        [
            "  output:",
            "    - type: s3msk",
            "      name: archive_audit_acceptance",
            "      src: /work/output",
            f"      dst: {json.dumps(output_dst)}",
            f"      bucket: {json.dumps(output_bucket)}",
            "      upload_policies:",
            "        - when: on_job_status=succeeded",
        ]
    )
    preset = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(preset, encoding="utf-8")
    return preset


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--verifier-uri", required=True)
    value.add_argument("--verifier-sha256", required=True)
    value.add_argument("--verifier-commit", required=True)
    value.add_argument("--report-uri", required=True)
    value.add_argument("--report-sha256", required=True)
    value.add_argument("--report-size-bytes", type=int, required=True)
    value.add_argument("--receipt-uri", required=True)
    value.add_argument("--receipt-sha256", required=True)
    value.add_argument("--metadata-uri", required=True)
    value.add_argument("--metadata-sha256", required=True)
    value.add_argument("--diagnostic-command-sha256", required=True)
    value.add_argument("--diagnostic-code-uri", required=True)
    value.add_argument("--source-f03-uri", required=True)
    value.add_argument("--source-f124-uri", required=True)
    value.add_argument("--diagnostic-output-prefix", required=True)
    value.add_argument("--output-prefix", required=True)
    value.add_argument("--region", required=True)
    value.add_argument("--image", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    build(parser().parse_args())


if __name__ == "__main__":
    main()
