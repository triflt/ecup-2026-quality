#!/usr/bin/env python3
"""Create a compact deterministic image bundle matching runtime pixel limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

MAX_PIXELS = 262144
JPEG_QUALITY = 85


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(*, runtime: Path, input_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    rows = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 40 or len({str(row["id"]) for row in rows}) != 40:
        raise ValueError("runtime must contain 40 unique rows")
    expected = {f"{row['id']}.jpg" for row in rows}
    actual = {path.name for path in input_dir.glob("*.jpg")}
    if actual != expected:
        raise ValueError("input images do not exactly match runtime")
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for filename in sorted(expected):
        source = input_dir / filename
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        if width * height > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / (width * height))
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            resized = image.resize(
                (max(28, int(width * scale)), max(28, int(height * scale))), resampling
            )
            image.close()
            image = resized
        destination = output_dir / filename
        image.save(
            destination,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
            progressive=False,
            subsampling=2,
        )
        final_size = image.size
        image.close()
        files[filename] = {
            "source_sha256": sha256_file(source),
            "output_sha256": sha256_file(destination),
            "output_bytes": destination.stat().st_size,
            "width": final_size[0],
            "height": final_size[1],
        }
    report = {
        "schema_version": "exp672_model_images_v1",
        "rows": len(rows),
        "runtime_sha256": sha256_file(runtime),
        "max_pixels": MAX_PIXELS,
        "jpeg_quality": JPEG_QUALITY,
        "jpeg_subsampling": 2,
        "files": files,
    }
    report_path = output_dir.parent / "model_image_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": len(rows),
        "total_output_bytes": sum(int(item["output_bytes"]) for item in files.values()),
        "report_sha256": sha256_file(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
