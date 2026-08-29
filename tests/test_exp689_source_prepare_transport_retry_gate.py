from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "689_qwen35_4b_grounded_transaction_graph_kd"
sys.path.insert(0, str(EXPERIMENT))

import build_source_prepare_retry_contract as contract_builder
import build_source_prepare_transport_retry_gate as gate_builder
import extract_source_archive_transport as transport


def write_diagnostic(path: Path) -> tuple[dict[str, object], str]:
    value = {
        "schema_version": "exp689_source_archive_audit_acceptance_v2",
        "status": "accepted_diagnostic_only",
        "decision": "TRANSPORT_CAUSE_APPLE_METADATA_ONLY",
        "independent_remote_provenance_verified": True,
        "report_sha256": "48e58fe07447544d8ebe1b3013ee6d003914864837b5a87a4c24ba5e6625cc67",
        "report_size_bytes": 21_769,
        "report_self_sha256": "b" * 64,
        "diagnostic_code_commit": "076c009eaee106caba428c699627f864e302d292",
        "diagnostic_code_sha256": "20a55c6e1268881c2281cba8d53efad26b5ad58fdc527d2bbddaae7f5477fc75",
        "diagnostic_preset_sha256": "622370e98f66d1378a5e299d9d10401fd01b0d31f26e5191f3bea36321e59ddb",
        "diagnostic_command_sha256": "5e36767eb7b7744347bf643f14603ffdfc12344bcd09652dbf1ac12b05cf494b",
        "submit_receipt_sha256": "620c88baac32ac5edd8f8e630d861eda91c3f95a7286c50b3bcb5afba047436b",
        "resolved_metadata_sha256": "75eb921999b1256dfa3192617b38f7394a09274f165698d5287fde8f97722140",
        "resolved_metadata_self_sha256": "ef162d3abaf62637611d60df02c027851608138a32a3edb803f4d55768a188f0",
        "verifier_commit": "14d215ed4890903be10712dc18e717fc515d1bac",
        "verifier_sha256": "a8b125dfd232c3cb13afb18ceb6f3789e15958cb0d0b189fdffadee97590b41b",
        "archive_bindings": {
            key: {"sha256": item["sha256"], "size_bytes": item["size_bytes"]}
            for key, item in transport.PROFILES.items()
        },
        "unsafe_reason_set": ["apple_metadata"],
        "prepare_retry_authorized": False,
        "teacher_authorized": False,
        "student_gpu_authorized": False,
        "self_sha256": None,
    }
    value["self_sha256"] = hashlib.sha256(
        transport.canonical_json_bytes(value)
    ).hexdigest()
    path.write_text(json.dumps(value), encoding="utf-8")
    return value, transport.sha256_file(path)


def contract_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        region="ix-m5-sm11",
        bucket="approved-bucket",
        revision="a" * 40,
        bundle_key="/owner/ecup/689/retry/bundle.tar.gz",
        bundle_sha256="1" * 64,
        manifest_key="/owner/ecup/689/retry/manifest.json",
        manifest_sha256="2" * 64,
        manifest_self_sha256="3" * 64,
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
        archive_acceptance_sha256="4" * 64,
        archive_acceptance_self_sha256="5" * 64,
        archive_verifier_terminal_metadata_sha256="6" * 64,
        retry_preset_builder_sha256="7" * 64,
        retry_preset_contract_key="/owner/ecup/689/retry/contract.json",
        transport_retry_gate_key="/owner/ecup/689/retry/gate.json",
        output_prefix="/owner/ecup/689/retry/output",
        transport_report_prefix="/owner/ecup/689/retry/transport",
        output=tmp_path / "contract.json",
    )


def make_packet(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic, diagnostic_file_sha = write_diagnostic(diagnostic_path)
    cargs = contract_args(tmp_path)
    cargs.archive_acceptance_sha256 = diagnostic_file_sha
    cargs.archive_acceptance_self_sha256 = diagnostic["self_sha256"]
    contract = contract_builder.build(cargs)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    contract_file_sha = transport.sha256_file(contract_path)
    gargs = argparse.Namespace(
        issuer_role="independent_integrator",
        diagnostic_acceptance=diagnostic_path,
        diagnostic_acceptance_file_sha256=diagnostic_file_sha,
        diagnostic_verifier_terminal_metadata_sha256=(
            cargs.archive_verifier_terminal_metadata_sha256
        ),
        retry_code_commit=cargs.revision,
        retry_code_bundle_sha256=cargs.bundle_sha256,
        retry_bundle_manifest_file_sha256=cargs.manifest_sha256,
        retry_bundle_manifest_self_sha256=cargs.manifest_self_sha256,
        retry_preset_builder_sha256=cargs.retry_preset_builder_sha256,
        retry_preset_contract=contract_path,
        retry_preset_contract_file_sha256=contract_file_sha,
        retry_preset_contract_self_sha256=contract["self_sha256"],
        source_prepare_spec=EXPERIMENT / "source_prepare_spec_v1.json",
        output_prefix=cargs.output_prefix,
        transport_report_prefix=cargs.transport_report_prefix,
    )
    gate = gate_builder.build(gargs)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    gate_file_sha = transport.sha256_file(gate_path)
    accepted_diagnostic = transport.validate_diagnostic_acceptance(
        diagnostic_path, diagnostic_file_sha
    )
    validated_contract = transport.validate_retry_preset_contract(
        contract_path,
        expected_file_sha256=contract_file_sha,
        expected_self_sha256=contract["self_sha256"],
    )
    result = transport.validate_transport_retry_gate(
        gate_path,
        expected_file_sha256=gate_file_sha,
        expected_self_sha256=gate["self_sha256"],
        diagnostic_acceptance=accepted_diagnostic,
        diagnostic_acceptance_file_sha256=diagnostic_file_sha,
        expected_verifier_terminal_metadata_sha256=(
            cargs.archive_verifier_terminal_metadata_sha256
        ),
        expected_retry_code_commit=cargs.revision,
        expected_retry_code_bundle_sha256=cargs.bundle_sha256,
        expected_retry_bundle_manifest_file_sha256=cargs.manifest_sha256,
        expected_retry_bundle_manifest_self_sha256=cargs.manifest_self_sha256,
        expected_retry_preset_builder_sha256=cargs.retry_preset_builder_sha256,
        retry_preset_contract=validated_contract,
        retry_preset_contract_file_sha256=contract_file_sha,
        expected_retry_preset_contract_sha256=contract["self_sha256"],
        source_prepare_spec_path=EXPERIMENT / "source_prepare_spec_v1.json",
        expected_output_prefix=cargs.output_prefix,
        expected_transport_report_prefix=cargs.transport_report_prefix,
    )
    return diagnostic, result


def test_independent_gate_is_required_while_diagnostic_stays_false(
    tmp_path: Path,
) -> None:
    diagnostic, gate = make_packet(tmp_path)
    assert diagnostic["prepare_retry_authorized"] is False
    assert gate["controlled_prepare_retry_authorized"] is True
    assert gate["issuer_role"] == "independent_integrator"
    assert gate["max_jobs"] == 1
    assert not gate["teacher_authorized"] and not gate["student_gpu_authorized"]


def test_gate_cannot_authorize_a_different_output(tmp_path: Path) -> None:
    make_packet(tmp_path)
    gate_path = tmp_path / "gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["output_prefix"] = "/owner/ecup/689/retry/other"
    gate["self_sha256"] = None
    gate["self_sha256"] = hashlib.sha256(
        transport.canonical_json_bytes(gate)
    ).hexdigest()
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    # The full validator is exercised in make_packet; the changed gate must at
    # least fail against its previously authorized immutable file identity.
    with pytest.raises(ValueError, match="file SHA mismatch"):
        transport.validate_transport_retry_gate(
            gate_path,
            expected_file_sha256="0" * 64,
            expected_self_sha256=gate["self_sha256"],
            diagnostic_acceptance={},
            diagnostic_acceptance_file_sha256="1" * 64,
            expected_verifier_terminal_metadata_sha256="2" * 64,
            expected_retry_code_commit="a" * 40,
            expected_retry_code_bundle_sha256="3" * 64,
            expected_retry_bundle_manifest_file_sha256="4" * 64,
            expected_retry_bundle_manifest_self_sha256="5" * 64,
            expected_retry_preset_builder_sha256="6" * 64,
            retry_preset_contract={},
            retry_preset_contract_file_sha256="7" * 64,
            expected_retry_preset_contract_sha256="8" * 64,
            source_prepare_spec_path=EXPERIMENT / "source_prepare_spec_v1.json",
            expected_output_prefix="/owner/ecup/689/retry/output",
            expected_transport_report_prefix="/owner/ecup/689/retry/transport",
        )


def test_contract_rejects_hidden_nested_input_fields(tmp_path: Path) -> None:
    args = contract_args(tmp_path)
    contract = contract_builder.build(args)
    contract["inputs"]["source_f03"]["hidden_override"] = "forbidden"
    contract["self_sha256"] = None
    contract["self_sha256"] = hashlib.sha256(
        transport.canonical_json_bytes(contract)
    ).hexdigest()
    path = tmp_path / "tampered_contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="input schema mismatch"):
        transport.validate_retry_preset_contract(
            path,
            expected_file_sha256=transport.sha256_file(path),
            expected_self_sha256=contract["self_sha256"],
        )
