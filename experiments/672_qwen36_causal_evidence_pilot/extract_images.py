#!/usr/bin/env python3
"""Extract only frozen first images from the canonical local archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(*, runtime: Path, images_zip: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    rows = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 40 or len({str(row["id"]) for row in rows}) != 40:
        raise ValueError("runtime must contain 40 unique rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(images_zip) as archive:
        names = set(archive.namelist())
        for row in rows:
            row_id = str(row["id"])
            member = str(row["image_member"])
            if member != f"images/{row_id}/0.jpg" or member not in names:
                raise ValueError(f"unsafe or missing image member for row {row_id}")
            destination = output_dir / f"{row_id}.jpg"
            with archive.open(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            files[row_id] = {
                "filename": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
    report = {
        "schema_version": "exp672_image_extract_v1",
        "rows": len(rows),
        "runtime_sha256": sha256_file(runtime),
        "images_zip_sha256": sha256_file(images_zip),
        "files": files,
    }
    report_path = output_dir.parent / "image_extract_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {key: value for key, value in report.items() if key != "files"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(extract(**vars(parser.parse_args())), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
