#!/usr/bin/env python3
"""Assemble an immutable local remote compute input bundle for experiment 672."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(*, runtime: Path, images_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    rows = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 40:
        raise ValueError("runtime must contain 40 rows")
    expected_images = {f"{row['id']}.jpg" for row in rows}
    actual_images = {path.name for path in images_dir.glob("*.jpg")}
    if actual_images != expected_images:
        raise ValueError("image directory does not exactly match runtime")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime, output_dir / "pilot_runtime.jsonl")
    shutil.copy2(HERE / "run_explanations.py", output_dir / "run_explanations.py")
    shutil.copy2(HERE / "frozen_spec.json", output_dir / "frozen_spec.json")
    target_images = output_dir / "images"
    target_images.mkdir()
    for source in sorted(images_dir.glob("*.jpg")):
        shutil.copy2(source, target_images / source.name)
    files = {
        str(path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "exp672_bundle_manifest_v1",
        "files": files,
        "rows": len(rows),
        "images": len(expected_images),
    }
    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": len(rows),
        "images": len(expected_images),
        "files": len(files) + 1,
        "bundle_manifest_sha256": sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
