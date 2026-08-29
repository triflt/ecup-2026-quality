from __future__ import annotations

import argparse
import hashlib
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
import extract_source_archive_transport as transport


def arguments(tmp_path: Path) -> argparse.Namespace:
    args = argparse.Namespace(
        region="ix-m5-sm11",
        bucket="approved-bucket",
        revision="a" * 40,
        bundle_key="/owner/ecup/689/retry/bundle.tar.gz",
        bundle_sha256="1" * 64,
        manifest_key="/owner/ecup/689/retry/manifest.json",
        manifest_sha256="2" * 64,
        manifest_self_sha256="a" * 64,
        source_f03_key="/owner/ecup/source/f03.tar.gz",
        source_f03_sha256=transport.PROFILES["source_f03"]["sha256"],
        source_f124_key="/owner/ecup/source/f124.tar.gz",
        source_f124_sha256=transport.PROFILES["source_f124"]["sha256"],
        exclusion_670_key="/owner/ecup/689/exp670.csv",
        exclusion_670_sha256=transport.EXCLUSION_BINDINGS[
            "exp670_audit_csv_sha256"
        ],
        exclusion_672_key="/owner/ecup/689/exp672.json",
        exclusion_672_sha256=transport.EXCLUSION_BINDINGS[
            "exp672_private_manifest_sha256"
        ],
        archive_acceptance_key="/owner/ecup/689/archive/acceptance.json",
        archive_acceptance_sha256="7" * 64,
        archive_acceptance_self_sha256="8" * 64,
        archive_verifier_terminal_metadata_sha256="9" * 64,
        transport_retry_gate_key="/owner/ecup/689/retry/gate.json",
        transport_retry_gate_sha256="b" * 64,
        transport_retry_gate_self_sha256="c" * 64,
        retry_preset_builder_sha256="d" * 64,
        retry_preset_contract_key="/owner/ecup/689/retry/contract.json",
        retry_preset_contract_file_sha256="e" * 64,
        retry_preset_contract_sha256="f" * 64,
        output_prefix="/owner/ecup/689/retry/output",
        transport_report_prefix="/owner/ecup/689/retry/transport",
        output=tmp_path / "preset.yaml",
        materialization_receipt_output=tmp_path / "receipt.json",
    )
    args.retry_preset_contract_sha256 = builder.preset_semantic_sha256(args)
    return args


def test_retry_preset_changes_only_safe_source_transport(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    text = builder.build(args)
    assert "flavor: 8cpu-128ram" in text
    assert text.count("    - type: s3msk") == 11
    assert "extract_source_archive_transport.py" in text
    assert "validate_diagnostic_acceptance" in text
    assert "validate_transport_retry_gate" in text
    assert "validate_retry_preset_contract" in text
    assert args.transport_retry_gate_sha256 in text
    assert args.transport_retry_gate_self_sha256 in text
    receipt = builder.materialization_receipt(args, text)
    assert receipt["final_preset_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert receipt["transport_retry_gate_file_sha256"] == (
        args.transport_retry_gate_sha256
    )
    assert receipt["max_jobs"] == 1
    assert "prepare_source_universe.py" in text
    assert "run_teacher" not in text and "student" not in text.lower()
    assert "access_key" not in text and "secret_key" not in text


def test_materialization_changes_when_gate_identity_changes(tmp_path: Path) -> None:
    first = arguments(tmp_path)
    first_text = builder.build(first)
    first_receipt = builder.materialization_receipt(first, first_text)
    second = arguments(tmp_path)
    second.transport_retry_gate_sha256 = "0" * 64
    second_text = builder.build(second)
    second_receipt = builder.materialization_receipt(second, second_text)
    assert first_text != second_text
    assert first_receipt["final_preset_sha256"] != second_receipt[
        "final_preset_sha256"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region", "unapproved"),
        ("archive_acceptance_sha256", "latest"),
        ("archive_acceptance_key", "relative/key"),
        ("transport_retry_gate_sha256", "latest"),
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
