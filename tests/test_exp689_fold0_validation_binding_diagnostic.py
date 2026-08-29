from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "689_qwen35_4b_grounded_transaction_graph_kd"
sys.path.insert(0, str(EXPERIMENT))

import build_fold0_validation_diagnostic_bundle as bundle_builder
import build_fold0_validation_diagnostic_preset as preset_builder
import diagnose_fold0_validation_binding as diagnostic
import extract_source_archive_transport as transport
import verify_fold0_validation_binding as independent


def _preset_args() -> object:
    return type(
        "Args",
        (),
        {
            "region": "ix-m5-sm11",
            "bucket": "approved-bucket",
            "revision": "a" * 40,
            "bundle_key": "/team/689/fold0_diag/code/bundle.tar.gz",
            "bundle_sha256": "1" * 64,
            "manifest_key": "/team/689/fold0_diag/code/manifest.json",
            "manifest_sha256": "2" * 64,
            "manifest_self_sha256": "3" * 64,
            "source_f03_key": "/team/689/source/f03.tar.gz",
            "source_f03_sha256": transport.PROFILES["source_f03"]["sha256"],
            "source_f03_size_bytes": transport.PROFILES["source_f03"][
                "size_bytes"
            ],
            "output_prefix": "/team/689/fold0_diag/output/unique",
            "output": Path("unused.yaml"),
        },
    )()


def _add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _write_bundle(tmp_path: Path) -> tuple[Path, Path, str, str]:
    bundle = tmp_path / "bundle"
    for relative in sorted(diagnostic.BUNDLE_PATHS):
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    revision = "a" * 40
    files = [
        {
            "path": relative,
            "sha256": diagnostic.sha256_file(bundle / relative),
            "size_bytes": (bundle / relative).stat().st_size,
        }
        for relative in sorted(diagnostic.BUNDLE_PATHS)
    ]
    manifest = {
        "schema_version": diagnostic.BUNDLE_SCHEMA,
        "builder_revision": revision,
        "files": files,
        "self_sha256": None,
    }
    manifest["self_sha256"] = hashlib.sha256(
        diagnostic.canonical_json_bytes(manifest)
    ).hexdigest()
    path = tmp_path / "bundle_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return bundle, path, revision, manifest["self_sha256"]


def _archive(
    tmp_path: Path,
    *,
    declared_validation_sha256: str,
    substitute_paths: bool = False,
) -> tuple[Path, dict[str, object]]:
    spec = json.loads((EXPERIMENT / "source_prepare_spec_v1.json").read_text())
    validation = b"\xffopaque-not-jsonl\x00\n"
    audit = {
        "experiment_id": "641",
        "outer_fold": 0,
        "source_runtime_contract_sha256": spec["folds"]["0"][
            "source_runtime_contract_sha256"
        ],
        "output_sha256": {"validation.jsonl": declared_validation_sha256},
    }
    audit["contract_sha256"] = hashlib.sha256(
        diagnostic.canonical_json_bytes(audit)
    ).hexdigest()
    prefix = "runtime/fold0" if substitute_paths else diagnostic.RUNTIME_ROOT.as_posix()
    path = tmp_path / ("substitute.tar" if substitute_paths else "source_f03.tar")
    with tarfile.open(path, "w") as archive:
        _add(
            archive,
            f"{prefix}/runtime_audit.json",
            json.dumps(audit).encode("utf-8"),
        )
        _add(archive, f"{prefix}/validation.jsonl", validation)
    profile: dict[str, object] = {
        "sha256": diagnostic.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "member_count": 2,
        "type_counts": {"regular": 2},
        "apple_metadata_count": 0,
    }
    return path, profile


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    spec = json.loads((EXPERIMENT / "source_prepare_spec_v1.json").read_text())
    archive, profile = _archive(
        tmp_path,
        declared_validation_sha256=spec["folds"]["0"]["validation_sha256"],
    )
    monkeypatch.setitem(transport.PROFILES, "source_f03", profile)
    monkeypatch.setattr(diagnostic, "_load_module", lambda _path: transport)
    monkeypatch.setattr(independent, "_module", lambda _path: transport)
    bundle, manifest, revision, manifest_self = _write_bundle(tmp_path)
    original_read_text = Path.read_text

    def reject_validation_parse(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "validation.jsonl":
            raise AssertionError("validation payload must remain opaque")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_validation_parse)
    report_path = tmp_path / "diagnostic.json"
    report = diagnostic.diagnose(
        archive_path=archive,
        bundle_root=bundle,
        bundle_sha256="b" * 64,
        manifest_path=manifest,
        manifest_sha256=diagnostic.sha256_file(manifest),
        manifest_self_sha256=manifest_self,
        revision=revision,
        work_dir=tmp_path / "work",
        output_path=report_path,
    )
    context: dict[str, object] = {
        "report_path": report_path,
        "archive_path": archive,
        "bundle_root": bundle,
        "bundle_sha256": "b" * 64,
        "manifest_path": manifest,
        "manifest_sha256": diagnostic.sha256_file(manifest),
        "manifest_self_sha256": manifest_self,
        "revision": revision,
    }
    return report, context


def test_declared_actual_mismatch_is_reported_without_payload_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _ = _run(tmp_path, monkeypatch)
    assert report["classification"] == (
        "ACTUAL_VALIDATION_DIFFERS_FROM_DECLARED_AND_FROZEN"
    )
    assert report["equality"] == {
        "declared_equals_actual": False,
        "declared_equals_frozen": True,
        "actual_equals_frozen": False,
        "source_runtime_contract_equals_frozen": True,
        "runtime_audit_contract_valid": True,
        "runtime_audit_self_valid": None,
    }
    assert report["labels_read"] == report["rows_read"] == 0
    assert report["retry_authorized"] is False


def test_exact_fold0_path_substitution_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = json.loads((EXPERIMENT / "source_prepare_spec_v1.json").read_text())
    archive, profile = _archive(
        tmp_path,
        declared_validation_sha256=spec["folds"]["0"]["validation_sha256"],
        substitute_paths=True,
    )
    monkeypatch.setitem(transport.PROFILES, "source_f03", profile)
    monkeypatch.setattr(diagnostic, "_load_module", lambda _path: transport)
    bundle, manifest, revision, manifest_self = _write_bundle(tmp_path)
    with pytest.raises(ValueError, match="missing or substituted"):
        diagnostic.diagnose(
            archive_path=archive,
            bundle_root=bundle,
            bundle_sha256="b" * 64,
            manifest_path=manifest,
            manifest_sha256=diagnostic.sha256_file(manifest),
            manifest_self_sha256=manifest_self,
            revision=revision,
            work_dir=tmp_path / "work",
            output_path=tmp_path / "diagnostic.json",
        )


def test_frozen_extraction_profile_remains_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = json.loads((EXPERIMENT / "source_prepare_spec_v1.json").read_text())
    archive, profile = _archive(
        tmp_path,
        declared_validation_sha256=spec["folds"]["0"]["validation_sha256"],
    )
    profile["member_count"] = 3
    monkeypatch.setitem(transport.PROFILES, "source_f03", profile)
    monkeypatch.setattr(diagnostic, "_load_module", lambda _path: transport)
    bundle, manifest, revision, manifest_self = _write_bundle(tmp_path)
    with pytest.raises(ValueError, match="member count mismatch"):
        diagnostic.diagnose(
            archive_path=archive,
            bundle_root=bundle,
            bundle_sha256="b" * 64,
            manifest_path=manifest,
            manifest_sha256=diagnostic.sha256_file(manifest),
            manifest_self_sha256=manifest_self,
            revision=revision,
            work_dir=tmp_path / "work",
            output_path=tmp_path / "diagnostic.json",
        )


def test_independent_verifier_recomputes_path_and_hash_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, context = _run(tmp_path, monkeypatch)
    result = independent.verify(
        **context,
        report_sha256=diagnostic.sha256_file(context["report_path"]),
        report_self_sha256=report["self_sha256"],
        work_dir=tmp_path / "verify-work",
    )
    assert result["classification"] == report["classification"]
    report["paths"]["validation"]["archive_member"] = "runtime/fold0/validation.jsonl"
    report["self_sha256"] = None
    report["self_sha256"] = hashlib.sha256(
        diagnostic.canonical_json_bytes(report)
    ).hexdigest()
    context["report_path"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="independent recomputation"):
        independent.verify(
            **context,
            report_sha256=diagnostic.sha256_file(context["report_path"]),
            report_self_sha256=report["self_sha256"],
            work_dir=tmp_path / "verify-work-tampered",
        )


def test_diagnostic_bundle_whitelist_is_exact_and_prepare_free() -> None:
    assert bundle_builder.FILES == tuple(sorted(diagnostic.BUNDLE_PATHS))
    assert {Path(path).name for path in bundle_builder.FILES} == {
        "diagnose_fold0_validation_binding.py",
        "extract_source_archive_transport.py",
        "source_prepare_spec_v1.json",
    }
    assert all("prepare_source_universe.py" not in path for path in bundle_builder.FILES)


def test_diagnostic_preset_is_cpu_only_three_input_single_output() -> None:
    preset = preset_builder.build(_preset_args())
    assert "flavor: 8cpu-128ram" in preset
    assert "gpu" not in preset.lower()
    assert preset.count("    - type: s3msk") == 4
    assert "      name: fold0_bind_report" in preset
    assert len("fold0_bind_report") <= 20
    assert "source_f124" not in preset
    assert "prepare_source_universe" not in preset
    assert "teacher" not in preset.lower()
    assert "student" not in preset.lower()
    assert "--source-f03-size-bytes" not in preset
    assert "stat -c%s /work/input/source_f03/source_f03.tar.gz" in preset
    assert "sha256sum /work/input/code/" in preset


def test_diagnostic_preset_rejects_source_or_output_substitution() -> None:
    args = _preset_args()
    args.source_f03_size_bytes += 1
    with pytest.raises(ValueError, match="frozen profile"):
        preset_builder.build(args)
    args = _preset_args()
    args.output_prefix = args.source_f03_key + "/nested"
    with pytest.raises(ValueError, match="disjoint"):
        preset_builder.build(args)


def test_diagnostic_cli_names_bind_exact_diagnose_signature() -> None:
    parsed = diagnostic.parser().parse_args(
        [
            "--archive",
            "source.tar.gz",
            "--bundle-root",
            "bundle",
            "--bundle-sha256",
            "1" * 64,
            "--manifest",
            "manifest.json",
            "--manifest-sha256",
            "2" * 64,
            "--manifest-self-sha256",
            "3" * 64,
            "--revision",
            "a" * 40,
            "--work-dir",
            "work",
            "--output",
            "report.json",
        ]
    )
    assert set(vars(parsed)) == {
        "archive_path",
        "bundle_root",
        "bundle_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_self_sha256",
        "revision",
        "work_dir",
        "output_path",
    }
    assert parsed.archive_path == Path("source.tar.gz")
    assert parsed.manifest_path == Path("manifest.json")
    assert parsed.output_path == Path("report.json")
