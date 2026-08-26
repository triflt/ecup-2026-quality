from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "689_qwen35_4b_grounded_transaction_graph_kd"
sys.path.insert(0, str(EXPERIMENT))

import build_source_prepare_retry_verifier_bundle as bundle_builder
import build_source_prepare_retry_verifier_preset as preset_builder
import extract_source_archive_transport as transport
import verify_source_prepare_retry as verifier


def preset_args() -> argparse.Namespace:
    return argparse.Namespace(
        region="ix-m5-sm11",
        bucket="approved-bucket",
        verifier_bundle_key="/team/689/verifier/code.tar.gz",
        verifier_bundle_sha256="1" * 64,
        verifier_manifest_key="/team/689/verifier/manifest.json",
        verifier_manifest_sha256="2" * 64,
        verifier_manifest_self_sha256="3" * 64,
        verifier_revision="a" * 40,
        verifier_runner_sha256="4" * 64,
        retry_bundle_key="/team/689/retry/code.tar.gz",
        retry_bundle_sha256="5" * 64,
        retry_manifest_key="/team/689/retry/manifest.json",
        retry_manifest_sha256="6" * 64,
        retry_manifest_self_sha256="7" * 64,
        retry_revision="b" * 40,
        extractor_sha256="8" * 64,
        source_f03_key="/team/689/source/f03.tar.gz",
        source_f03_sha256=transport.PROFILES["source_f03"]["sha256"],
        source_f124_key="/team/689/source/f124.tar.gz",
        source_f124_sha256=transport.PROFILES["source_f124"]["sha256"],
        exclusion_670_key="/team/689/exclusion/670.csv",
        exclusion_670_sha256=transport.EXCLUSION_BINDINGS[
            "exp670_audit_csv_sha256"
        ],
        exclusion_672_key="/team/689/exclusion/672.json",
        exclusion_672_sha256=transport.EXCLUSION_BINDINGS[
            "exp672_private_manifest_sha256"
        ],
        prepared_prefix="/team/689/prepare/retry1",
        prepare_transport_report_prefix="/team/689/prepare-transport/retry1",
        diagnostic_acceptance_key="/team/689/diagnostic/acceptance.json",
        diagnostic_acceptance_sha256="a" * 64,
        diagnostic_acceptance_self_sha256="b" * 64,
        diagnostic_verifier_terminal_metadata_sha256="c" * 64,
        retry_contract_key="/team/689/retry/contract.json",
        retry_contract_sha256="d" * 64,
        retry_contract_self_sha256="e" * 64,
        retry_gate_key="/team/689/retry/gate.json",
        retry_gate_sha256="f" * 64,
        retry_gate_self_sha256="0" * 64,
        retry_preset_builder_sha256="1" * 64,
        expected_resolved_command_sha256="e" * 64,
        submit_receipt_key="/team/689/submit/receipt.json",
        submit_receipt_sha256="2" * 64,
        submit_receipt_self_sha256="3" * 64,
        live_go_key="/team/689/go/live_go.json",
        live_go_sha256="4" * 64,
        live_go_self_sha256="5" * 64,
        materialization_receipt_key="/team/689/materialization/receipt.json",
        materialization_receipt_sha256="6" * 64,
        materialization_receipt_self_sha256="7" * 64,
        resolved_terminal_metadata_key="/team/689/terminal/resolved.json",
        resolved_terminal_metadata_sha256="8" * 64,
        resolved_terminal_metadata_self_sha256="9" * 64,
        terminal_transport_f03_key=(
            "/team/689/prepare-transport/retry1/source_f03_extraction.json"
        ),
        terminal_transport_f03_sha256="a" * 64,
        terminal_transport_f03_self_sha256="b" * 64,
        terminal_transport_f124_key=(
            "/team/689/prepare-transport/retry1/source_f124_extraction.json"
        ),
        terminal_transport_f124_sha256="c" * 64,
        terminal_transport_f124_self_sha256="d" * 64,
        output_prefix="/team/689/verification/retry1",
        output=Path("unused.yaml"),
    )


def test_transport_aware_preset_is_cpu_only_and_uses_original_archives() -> None:
    preset = preset_builder.build(preset_args())
    assert "flavor: 8cpu-128ram" in preset
    assert "gpu" not in preset.lower()
    assert "access_key" not in preset and "secret_key" not in preset
    assert "verify_source_prepare_retry.py" in preset
    assert "--retry-gate /work/input/retry_gate/transport_retry_gate.json" in preset
    assert "--diagnostic-acceptance " in preset
    assert "--submit-receipt /work/input/submit_receipt/submit_receipt.json" in preset
    assert "--live-go /work/input/live_go/live_go.json" in preset
    assert (
        "--materialization-receipt "
        "/work/input/materialization/materialization_receipt.json" in preset
    )
    assert (
        "--resolved-terminal-metadata "
        "/work/input/resolved_terminal/resolved_terminal_metadata.json" in preset
    )
    assert (
        "--terminal-transport-f03 "
        "/work/input/terminal_transport_f03/source_f03_extraction.json" in preset
    )
    assert (
        "--terminal-transport-f124 "
        "/work/input/terminal_transport_f124/source_f124_extraction.json" in preset
    )
    assert "--source-f03 /work/input/source_f03/source_f03.tar.gz" in preset
    assert "--source-f124 /work/input/source_f124/source_f124.tar.gz" in preset
    assert "--prepare-dir /work/input/prepared/prepared" in preset
    assert "--transport-report-dir /work/output/transport_reports" in preset
    assert "--retry-bundle-sha256 " + "5" * 64 in preset
    assert "--retry-manifest-self-sha256 " + "7" * 64 in preset
    assert "  input:\n" in preset and "  output:\n" in preset
    assert "  inputs:\n" not in preset and "  outputs:\n" not in preset
    assert "      name: src_retry_accept" in preset
    assert len("src_retry_accept") <= 20
    assert preset.count("    - type: s3msk") == 19


def test_transport_aware_preset_rejects_archive_or_output_substitution() -> None:
    args = preset_args()
    args.source_f03_sha256 = "2" * 64
    with pytest.raises(ValueError, match="frozen source archive"):
        preset_builder.build(args)
    args = preset_args()
    args.output_prefix = args.retry_gate_key + "/nested"
    with pytest.raises(ValueError, match="disjoint"):
        preset_builder.build(args)


@pytest.mark.parametrize(
    "input_field",
    [
        "verifier_bundle_key",
        "retry_bundle_key",
        "source_f03_key",
        "exclusion_670_key",
        "retry_gate_key",
        "resolved_terminal_metadata_key",
        "prepared_prefix",
        "terminal_transport_f03_key",
    ],
)
def test_verifier_output_is_disjoint_from_every_input_class(
    input_field: str,
) -> None:
    args = preset_args()
    args.output_prefix = getattr(args, input_field) + "/nested"
    with pytest.raises(ValueError, match="disjoint"):
        preset_builder.build(args)


def test_non_overlapping_rejects_two_verifier_outputs() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        preset_builder._non_overlapping(outputs=["/out", "/out/nested"], inputs=[])


def test_resolved_terminal_rejects_second_platform_attempt() -> None:
    with pytest.raises(ValueError, match="retry/platform attempt mismatch"):
        verifier._validate_terminal_attempts(
            {"retry_attempt": 1, "platform_attempt": 2}
        )


def test_retry_verifier_bundle_is_deterministic_and_runner_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in bundle_builder.FILES:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    first = bundle_builder.build(repo, revision, tmp_path / "first")
    second = bundle_builder.build(repo, revision, tmp_path / "second")
    assert first["archive_sha256"] == second["archive_sha256"]
    assert [item["path"] for item in first["files"]] == list(bundle_builder.FILES)
    assert first["teacher_code_members"] == first["student_code_members"] == 0


def _self_hash(value: dict[str, object]) -> dict[str, object]:
    value["self_sha256"] = None
    value["self_sha256"] = hashlib.sha256(
        verifier.canonical_json_bytes(value)
    ).hexdigest()
    return value


def test_transport_report_binding_rejects_sanitizer_claim_mutation(
    tmp_path: Path,
) -> None:
    profile = {
        "sha256": "1" * 64,
        "size_bytes": 10,
        "member_count": 3,
        "type_counts": {"directory": 1, "regular": 2},
        "apple_metadata_count": 1,
    }
    report = _self_hash(
        {
            "schema_version": "exp689_source_archive_transport_extraction_v1",
            "archive_id": "source_f03",
            "archive_sha256": profile["sha256"],
            "archive_size_bytes": 10,
            "member_count": 3,
            "type_counts": profile["type_counts"],
            "apple_metadata_skipped_count": 1,
            "apple_metadata_skipped_names": ["._validation.jsonl"],
            "extracted_regular_files": 1,
            "extracted_directories": 1,
            "symlinks_extracted": 0,
            "hardlinks_extracted": 0,
            "path_traversal_extracted": 0,
            "self_sha256": None,
        }
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    binding = verifier._validate_transport_report(path, "source_f03", profile)
    assert binding["report_self_sha256"] == report["self_sha256"]
    report["apple_metadata_skipped_count"] = 0
    _self_hash(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen result mismatch"):
        verifier._validate_transport_report(path, "source_f03", profile)


def test_gate_rejection_happens_before_original_archive_payload_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier_root = tmp_path / "verifier"
    retry_root = tmp_path / "retry"
    runner = verifier_root / verifier.RUNNER_PATH
    extractor = retry_root / (
        verifier.EXPERIMENT + "/extract_source_archive_transport.py"
    )
    runner.parent.mkdir(parents=True)
    extractor.parent.mkdir(parents=True)
    runner.write_text("# runner\n", encoding="utf-8")
    extractor.write_text("# extractor\n", encoding="utf-8")
    inputs: dict[str, Path] = {}
    for name in (
        "verifier_manifest",
        "retry_manifest",
        "source_f03",
        "source_f124",
        "exclusion_670",
        "exclusion_672",
        "diagnostic_acceptance",
        "retry_contract",
        "retry_gate",
        "submit_receipt",
        "live_go",
        "materialization_receipt",
        "resolved_terminal_metadata",
        "terminal_transport_f03",
        "terminal_transport_f124",
    ):
        path = tmp_path / f"{name}.dat"
        path.write_bytes(b"frozen")
        inputs[name] = path
    runner_sha = verifier.sha256_file(runner)
    extractor_sha = verifier.sha256_file(extractor)

    def manifest(**kwargs: object) -> dict[str, object]:
        if kwargs["expected_schema"].endswith("verifier_bundle_manifest_v1"):
            return {"files": [{"path": verifier.RUNNER_PATH, "sha256": runner_sha}]}
        return {
            "files": [
                {
                    "path": path,
                    "sha256": extractor_sha if path.endswith("extract_source_archive_transport.py") else "0" * 64,
                }
                for path in sorted(verifier.RETRY_PATHS)
            ]
        }

    extract = mock.Mock()

    def reject_gate(*_args: object, **_kwargs: object) -> None:
        raise ValueError("independent gate rejected")

    fake_transport = SimpleNamespace(
        validate_retry_preset_contract=lambda *_args, **_kwargs: {"self_sha256": "8" * 64},
        validate_diagnostic_acceptance=lambda *_args, **_kwargs: {"self_sha256": "7" * 64},
        validate_transport_retry_gate=reject_gate,
        extract=extract,
    )
    fake_source = SimpleNamespace()
    monkeypatch.setattr(verifier, "_validate_code_manifest", manifest)
    monkeypatch.setattr(
        verifier, "_load_module", mock.Mock(side_effect=[fake_transport, fake_source])
    )
    values: dict[str, object] = {
        **inputs,
        "verifier_bundle_root": verifier_root,
        "retry_bundle_root": retry_root,
        "verifier_bundle_sha256": "1" * 64,
        "verifier_manifest_sha256": "2" * 64,
        "verifier_manifest_self_sha256": "3" * 64,
        "verifier_runner_sha256": runner_sha,
        "retry_bundle_sha256": "4" * 64,
        "retry_manifest_sha256": "5" * 64,
        "retry_manifest_self_sha256": "6" * 64,
        "extractor_sha256": extractor_sha,
        "source_f03_sha256": "a" * 64,
        "source_f124_sha256": "b" * 64,
        "exclusion_670_sha256": "c" * 64,
        "exclusion_672_sha256": "d" * 64,
        "diagnostic_acceptance_sha256": "e" * 64,
        "diagnostic_acceptance_self_sha256": "7" * 64,
        "diagnostic_verifier_terminal_metadata_sha256": "f" * 64,
        "retry_contract_sha256": "1" * 64,
        "retry_contract_self_sha256": "8" * 64,
        "retry_gate_sha256": "2" * 64,
        "retry_gate_self_sha256": "3" * 64,
        "retry_preset_builder_sha256": "4" * 64,
        "expected_resolved_command_sha256": "1" * 64,
        "submit_receipt_sha256": "5" * 64,
        "submit_receipt_self_sha256": "6" * 64,
        "live_go_sha256": "7" * 64,
        "live_go_self_sha256": "8" * 64,
        "materialization_receipt_sha256": "9" * 64,
        "materialization_receipt_self_sha256": "a" * 64,
        "resolved_terminal_metadata_sha256": "b" * 64,
        "resolved_terminal_metadata_self_sha256": "c" * 64,
        "terminal_transport_f03_sha256": "d" * 64,
        "terminal_transport_f03_self_sha256": "e" * 64,
        "terminal_transport_f124_sha256": "f" * 64,
        "terminal_transport_f124_self_sha256": "0" * 64,
        "verifier_revision": "a" * 40,
        "retry_revision": "b" * 40,
        "base_acceptance": tmp_path / "out/base.json",
        "transport_report_dir": tmp_path / "out/reports",
        "runtime_root": tmp_path / "runtime",
        "acceptance": tmp_path / "out/final.json",
        "prepare_output_prefix": "/owner/prepare",
        "prepare_transport_report_prefix": "/owner/transport",
        "retry_bundle_ref": "s3://bucket/retry.tar.gz",
        "retry_manifest_ref": "s3://bucket/manifest.json",
        "source_f03_ref": "s3://bucket/f03.tar.gz",
        "source_f124_ref": "s3://bucket/f124.tar.gz",
        "exclusion_670_ref": "s3://bucket/670.csv",
        "exclusion_672_ref": "s3://bucket/672.json",
        "diagnostic_acceptance_ref": "s3://bucket/diagnostic.json",
        "retry_contract_ref": "s3://bucket/contract.json",
        "retry_gate_ref": "s3://bucket/gate.json",
        "submit_receipt_ref": "s3://bucket/submit.json",
        "live_go_ref": "s3://bucket/live_go.json",
        "materialization_receipt_ref": "s3://bucket/materialization.json",
        "resolved_terminal_metadata_ref": "s3://bucket/terminal.json",
        "terminal_transport_f03_ref": "s3://bucket/transport/source_f03_extraction.json",
        "terminal_transport_f124_ref": "s3://bucket/transport/source_f124_extraction.json",
        "prepare_output_ref": "s3://bucket/prepare",
        "prepare_dir": tmp_path / "prepared",
    }
    with pytest.raises(ValueError, match="independent gate rejected"):
        verifier.verify(argparse.Namespace(**values))
    extract.assert_not_called()
