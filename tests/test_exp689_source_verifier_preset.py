from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "689_qwen35_4b_grounded_transaction_graph_kd"
)
sys.path.insert(0, str(EXPERIMENT))

import build_source_prepare_verifier_preset as builder


def args() -> Namespace:
    return Namespace(
        region="ix-m5-sm11",
        bucket="approved-bucket",
        revision="a" * 40,
        bundle_key="/team/ecup/689/code/source_prepare.tar.gz",
        bundle_sha256="1" * 64,
        manifest_key="/team/ecup/689/code/bundle_manifest.json",
        manifest_sha256="2" * 64,
        source_f03_key="/team/ecup/689/source/f03.tar.gz",
        source_f03_sha256="3" * 64,
        source_f124_key="/team/ecup/689/source/f124.tar.gz",
        source_f124_sha256="4" * 64,
        exclusion_670_key="/team/ecup/689/exclusions/670.csv",
        exclusion_670_sha256="5" * 64,
        exclusion_672_key="/team/ecup/689/exclusions/672.json",
        exclusion_672_sha256="6" * 64,
        prepared_prefix="/team/ecup/689/source_prepare/run1",
        terminal_metadata_key="/team/ecup/689/source_prepare/terminal/run1.json",
        terminal_metadata_sha256="7" * 64,
        output_prefix="/team/ecup/689/source_prepare/acceptance/run1",
    )


def test_remote_first_verifier_preset_is_cpu_only_and_secret_free() -> None:
    preset = builder.build(args())
    assert "flavor: 8cpu-128ram" in preset
    assert "gpu" not in preset.lower()
    assert "access_key" not in preset
    assert "secret_key" not in preset
    assert "verify_source_prepare.py" in preset
    assert "--prepare-dir /work/input/prepared/prepared" in preset
    assert "--runtime-dir" in preset
    assert preset.count("--runtime-dir") == 5
    assert preset.count("--runtime-archive ") == 2
    assert "--terminal-metadata-sha256 " + "7" * 64 in preset
    assert (
        "--approved-s3-output-ref "
        "s3://approved-bucket/team/ecup/689/source_prepare/run1"
    ) in preset
    assert "teacher" not in preset.lower()
    assert "student" not in preset.lower()
    assert "  input:\n" in preset
    assert "  output:\n" in preset
    assert "  inputs:\n" not in preset
    assert "  outputs:\n" not in preset
    assert 'src: "/team/ecup/689/source_prepare/run1"' in preset
    assert 'dst: "/work/input/prepared"' in preset


def test_verifier_preset_rejects_unpinned_terminal_metadata() -> None:
    value = args()
    value.terminal_metadata_sha256 = "latest"
    with pytest.raises(ValueError, match="exact lowercase SHA-256"):
        builder.build(value)


def test_verifier_preset_rejects_relative_or_query_keys() -> None:
    value = args()
    value.prepared_prefix = "relative/path"
    with pytest.raises(ValueError, match="unsafe S3 key"):
        builder.build(value)
    value = args()
    value.prepared_prefix += "?signature=forbidden"
    with pytest.raises(ValueError, match="unsafe S3 key"):
        builder.build(value)
