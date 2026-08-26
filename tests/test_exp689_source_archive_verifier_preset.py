from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "689_qwen35_4b_grounded_transaction_graph_kd"
)
sys.path.insert(0, str(EXPERIMENT))

import build_source_archive_verifier_preset as builder


def arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        verifier_uri="s3://approved/exp689/code/verifier.py",
        verifier_sha256="a" * 64,
        verifier_commit="b" * 40,
        report_uri="s3://approved/exp689/diagnostic/archive_header_audit.json",
        report_sha256="c" * 64,
        report_size_bytes=1234,
        receipt_uri="s3://approved/exp689/diagnostic/submit_receipt.json",
        receipt_sha256="d" * 64,
        metadata_uri="s3://approved/exp689/diagnostic/resolved_metadata.json",
        metadata_sha256="e" * 64,
        diagnostic_command_sha256="f" * 64,
        diagnostic_code_uri="s3://approved/exp689/code/audit_source_archives.py",
        source_f03_uri="s3://approved/exp689/source/f03.tar.gz",
        source_f124_uri="s3://approved/exp689/source/f124.tar.gz",
        diagnostic_output_prefix="s3://approved/exp689/diagnostic",
        output_prefix="s3://approved/exp689/acceptance/run1",
        region="safe-region",
        image="safe/image:1.0",
        output=tmp_path / "preset.yaml",
    )


def test_builder_emits_cpu_only_exact_remote_verifier(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    preset = builder.build(args)
    assert "flavor: 8cpu-128ram" in preset
    assert preset.count("    - type: s3msk") == 5
    assert args.verifier_sha256 in preset
    assert args.report_sha256 in preset
    assert args.receipt_sha256 in preset
    assert args.metadata_sha256 in preset
    assert args.diagnostic_command_sha256 in preset
    assert "--approved-output-prefix s3://approved/exp689/diagnostic" in preset
    assert args.output.is_file()


@pytest.mark.parametrize(
    "bad_uri",
    [
        "s3://approved/key?token=secret",
        "s3://approved/key#fragment",
        "s3://approved/key\\tail",
        "s3://approved/../escape",
    ],
)
def test_builder_rejects_unsafe_s3_uri(tmp_path: Path, bad_uri: str) -> None:
    args = arguments(tmp_path)
    args.report_uri = bad_uri
    with pytest.raises(ValueError, match="forbidden URI syntax|unsafe S3 key"):
        builder.build(args)


def test_builder_rejects_unpinned_hash_or_overwrite(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    args.metadata_sha256 = "latest"
    with pytest.raises(ValueError, match="exact lowercase SHA-256"):
        builder.build(args)
    args = arguments(tmp_path)
    builder.build(args)
    with pytest.raises(FileExistsError, match="overwrite"):
        builder.build(args)
