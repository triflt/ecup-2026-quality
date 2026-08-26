from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "689_qwen35_4b_grounded_transaction_graph_kd"
sys.path.insert(0, str(EXPERIMENT))

import build_source_prepare_bundle as bundle
import build_source_prepare_preset as preset


def test_bundle_is_deterministic_and_contains_only_source_prepare(tmp_path: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "fd4a768"], text=True
    ).strip()
    first = bundle.build(ROOT, revision, tmp_path / "first")
    second = bundle.build(ROOT, revision, tmp_path / "second")
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["bundle_manifest_sha256"] == second["bundle_manifest_sha256"]
    assert [item["path"] for item in first["files"]] == list(bundle.FILES)
    assert first["teacher_code_members"] == first["student_code_members"] == 0


def make_args(tmp_path: Path) -> Namespace:
    return Namespace(
        region="ix-m5-sm11",
        bucket="approved-bucket",
        revision="a" * 40,
        bundle_key="/owner/ecup/689/code.tar.gz",
        bundle_sha256="1" * 64,
        manifest_key="/owner/ecup/689/manifest.json",
        manifest_sha256="2" * 64,
        source_f03_key="/owner/ecup/source/f03.tar.gz",
        source_f03_sha256="3" * 64,
        source_f124_key="/owner/ecup/source/f124.tar.gz",
        source_f124_sha256="4" * 64,
        exclusion_670_key="/owner/ecup/689/exp670.csv",
        exclusion_670_sha256="5" * 64,
        exclusion_672_key="/owner/ecup/689/exp672.json",
        exclusion_672_sha256="6" * 64,
        output_prefix="/owner/ecup/689/output/unique",
        output=tmp_path / "preset.yaml",
    )


def test_preset_is_secret_free_cpu_only_and_exact_scope(tmp_path: Path) -> None:
    text = preset.build(make_args(tmp_path))
    assert "flavor: 8cpu-128ram" in text
    assert "region: ix-m5-sm11" in text
    assert text.count("    - type: s3msk") == 7
    assert "access_key" not in text and "secret_key" not in text
    assert "run_teacher" not in text and "train" not in text.lower()
    assert "prepare_source_universe.py" in text
    assert "source_prepare_bundle.tar.gz" in text
    assert "on_job_status=succeeded" in text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region", "wrong-region"),
        ("bundle_key", "relative/path"),
        ("output_prefix", "/owner/ecup/../escape"),
        ("source_f03_sha256", "0" * 63),
    ],
)
def test_preset_rejects_unfrozen_transport(tmp_path: Path, field: str, value: str) -> None:
    args = make_args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        preset.build(args)


def test_bundle_builder_has_no_optional_dynamic_import_surface() -> None:
    source = (EXPERIMENT / "build_source_prepare_bundle.py").read_text(encoding="utf-8")
    assert importlib.util.find_spec("tarfile") is not None
    assert "eval(" not in source and "exec(" not in source
    json.loads(json.dumps({"files": bundle.FILES}))
