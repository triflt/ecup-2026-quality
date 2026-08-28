from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_path(cache: Path, row_id: str, architecture: str) -> Path:
    suffix = ".jpg" if architecture == "qwen3vl_2b" else ".img"
    return cache / f"{hashlib.sha256(row_id.encode()).hexdigest()}{suffix}"


def read_ids(
    runtime_root: Path, folds: tuple[int, ...]
) -> tuple[set[str], dict[str, str]]:
    ids: set[str] = set()
    sources = {}
    for fold in folds:
        for name in ("train.jsonl", "validation.jsonl"):
            path = runtime_root / f"fold{fold}" / name
            sources[f"fold{fold}/{name}"] = sha256(path)
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if not row.get("synthetic"):
                        ids.add(str(row["id"]))
    return ids, sources


def prepare(
    runtime_root: Path,
    image_root: Path,
    cache_root: Path,
    architecture: str,
    folds: tuple[int, ...] = (0, 3),
) -> dict:
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(5) for fold in folds):
        raise ValueError("invalid fold list")
    ids, sources = read_ids(runtime_root, folds)
    cache_root.mkdir(parents=True, exist_ok=True)
    created = 0
    reused = 0
    for row_id in sorted(ids):
        source = image_root / row_id / "0.jpg"
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"missing regular mounted first image for id {row_id}")
        destination = cache_path(cache_root, row_id, architecture)
        if destination.is_file():
            reused += 1
            continue
        if destination.exists() or destination.is_symlink():
            raise ValueError("invalid pre-existing image-cache entry")
        if architecture == "qwen35_4b":
            os.symlink(source, destination)
        else:
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                image.thumbnail((448, 448), Image.Resampling.LANCZOS)
                image.save(destination, format="JPEG", quality=92)
                image.close()
        created += 1
    report = {
        "schema": "exp699_remote_compute_image_cache_v1",
        "architecture": architecture,
        "folds": list(folds),
        "unique_ids": len(ids),
        "created": created,
        "reused": reused,
        "runtime_sources": sources,
        "mounted_image_root": str(image_root),
        "cache_root": str(cache_root),
        "all_entries_ready": created + reused == len(ids),
    }
    report["self_sha256"] = hashlib.sha256(
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("qwen35_4b", "qwen3vl_2b"), required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), action="append", default=[])
    args = parser.parse_args()
    report = prepare(
        args.runtime_root.resolve(),
        args.image_root.resolve(),
        args.cache_root.resolve(),
        args.architecture,
        tuple(args.fold or (0, 3)),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
