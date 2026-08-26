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

import build_source_prepare_retry_preset as builder


def arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        region="ix-m5-sm11",
        bucket="approved-bucket",
        revision="a" * 40,
        bundle_key="/owner/ecup/689/retry/bundle.tar.gz",
        bundle_sha256="1" * 64,
        manifest_key="/owner/ecup/689/retry/manifest.json",
        manifest_sha256="2" * 64,
        source_f03_key="/owner/ecup/source/f03.tar.gz",
        source_f03_sha256="3" * 64,
        source_f124_key="/owner/ecup/source/f124.tar.gz",
        source_f124_sha256="4" * 64,
        exclusion_670_key="/owner/ecup/689/exp670.csv",
        exclusion_670_sha256="5" * 64,
        exclusion_672_key="/owner/ecup/689/exp672.json",
        exclusion_672_sha256="6" * 64,
        archive_acceptance_key="/owner/ecup/689/archive/acceptance.json",
        archive_acceptance_sha256="7" * 64,
        output_prefix="/owner/ecup/689/retry/output",
        transport_report_prefix="/owner/ecup/689/retry/transport",
        output=tmp_path / "preset.yaml",
    )


def test_retry_preset_changes_only_safe_source_transport(tmp_path: Path) -> None:
    text = builder.build(arguments(tmp_path))
    assert "flavor: 8cpu-128ram" in text
    assert text.count("    - type: s3msk") == 9
    assert "extract_source_archive_transport.py" in text
    assert "validate_diagnostic_acceptance" in text
    assert "prepare_source_universe.py" in text
    assert "run_teacher" not in text and "student" not in text.lower()
    assert "access_key" not in text and "secret_key" not in text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region", "unapproved"),
        ("archive_acceptance_sha256", "latest"),
        ("archive_acceptance_key", "relative/key"),
        ("transport_report_prefix", "/owner/ecup/../escape"),
    ],
)
def test_retry_preset_rejects_unfrozen_transport(
    tmp_path: Path, field: str, value: str
) -> None:
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        builder.build(args)
