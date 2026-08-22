from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "research/scoped_runtime_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("scoped_runtime_bootstrap", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def _archive(path: Path, members: dict[str, bytes]) -> str:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_runtime_checks_hash_and_extracts_files(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.tar.gz"
    checksum = _archive(archive, {"development_data.csv": b"id,label\na,1\n"})
    output = tmp_path / "runtime"

    audit = bootstrap.prepare_runtime(
        url=archive.as_uri(), expected_sha256=checksum, output_dir=output
    )

    assert audit["archive_sha256"] == checksum
    assert audit["runtime_files"] == 1
    assert (output / "development_data.csv").read_bytes() == b"id,label\na,1\n"


def test_prepare_runtime_rejects_bad_hash_and_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.tar.gz"
    checksum = _archive(archive, {"development_data.csv": b"ok"})
    with pytest.raises(ValueError, match="checksum mismatch"):
        bootstrap.prepare_runtime(
            url=archive.as_uri(), expected_sha256="0" * 64, output_dir=tmp_path / "bad"
        )

    unsafe = tmp_path / "unsafe.tar.gz"
    unsafe_checksum = _archive(unsafe, {"../escape": b"no"})
    with pytest.raises(ValueError, match="unsafe archive member"):
        bootstrap.prepare_runtime(
            url=unsafe.as_uri(),
            expected_sha256=unsafe_checksum,
            output_dir=tmp_path / "unsafe-output",
        )
    assert checksum != unsafe_checksum
