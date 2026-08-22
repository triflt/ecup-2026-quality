from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path("/work/input")
OUTPUT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
SCORER = Path("/work/code/qwen35_flammable_adapter_soup_oof.py")
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract(name: str) -> Path:
    url = os.environ[f"SOURCE_{name}_URL"]
    expected = os.environ[f"SOURCE_{name}_SHA256"]
    archive_path = ROOT / f"source{name}.zip"
    destination = ROOT / f"source{name}"
    temporary = archive_path.with_suffix(".download")
    if archive_path.exists() or temporary.exists() or destination.exists():
        raise FileExistsError(f"refusing to reuse source path for {name}")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink()
        raise ValueError(f"source {name} checksum mismatch: {actual} != {expected}")
    temporary.rename(archive_path)
    total = 0
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"source {name} CRC failure: {bad}")
        for info in archive.infolist():
            path = Path(info.filename)
            mode = info.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                raise ValueError(f"unsafe source {name} member: {info.filename}")
            total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError(f"source {name} expands above safe limit: {total}")
        archive.extractall(destination)
    expected_files = {f"fold_{fold}.zip" for fold in range(5)}
    actual_files = {path.name for path in destination.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(f"source {name} file set mismatch: {sorted(actual_files)}")
    return destination


def main() -> None:
    source190 = download_and_extract("190")
    source260 = download_and_extract("260")
    environment = os.environ.copy()
    environment["SOURCE_190"] = str(source190)
    environment["SOURCE_260"] = str(source260)
    subprocess.run([sys.executable, "-u", str(SCORER)], env=environment, check=True)
    suffix = os.environ["BUNDLE_SUFFIX"]
    archive = Path(shutil.make_archive(f"/work/qwen_soup_{suffix}", "zip", OUTPUT))
    destination = OUTPUT / f"output_bundle_{suffix}.zip"
    shutil.move(str(archive), destination)
    print(f"saved output bundle: {destination}", flush=True)


if __name__ == "__main__":
    main()
