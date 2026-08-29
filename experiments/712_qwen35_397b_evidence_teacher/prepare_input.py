from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
EXPECTED_IMAGES_SHA256 = "9ff051272e9ee1b08d6d05939470ebc7006403f4647c546b29647efb03837edc"
EXPECTED_IMAGE_FILES = 49_456


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"existing file has unexpected hash: {destination}")
        return
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial download: {partial}")
    urllib.request.urlretrieve(url, partial)
    actual = sha256_file(partial)
    if actual != expected_sha256:
        raise ValueError(f"download hash mismatch for {destination}: {actual}")
    partial.replace(destination)


def safe_extract(archive: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = [member for member in source.infolist() if not member.is_dir()]
        images = [
            member for member in members
            if member.filename.startswith("images/") and member.filename.lower().endswith(".jpg")
        ]
        if len(images) != EXPECTED_IMAGE_FILES:
            raise ValueError(f"expected {EXPECTED_IMAGE_FILES} images, got {len(images)}")
        root_resolved = root.resolve()
        for member in images:
            destination = (root / member.filename).resolve()
            if root_resolved not in destination.parents:
                raise ValueError(f"unsafe ZIP member: {member.filename}")
        for index, member in enumerate(images, 1):
            source.extract(member, root)
            if index % 5000 == 0 or index == len(images):
                print(f"extracted_images={index}/{len(images)}", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: prepare_input.py COMMAND [ARGS...]")
    images_url = os.environ.get("IMAGES_ZIP_URL")
    data_url = os.environ.get("DATA_CSV_URL")
    if not images_url or not data_url:
        raise ValueError("IMAGES_ZIP_URL and DATA_CSV_URL must be injected at submission time")
    cache = Path("/work/cache")
    input_root = Path("/work/input")
    images_zip = cache / "images.zip"
    data_csv = input_root / "data.csv"
    download(data_url, data_csv, EXPECTED_DATA_SHA256)
    download(images_url, images_zip, EXPECTED_IMAGES_SHA256)
    safe_extract(images_zip, input_root)
    subprocess.run(sys.argv[1:], check=True)


if __name__ == "__main__":
    main()
