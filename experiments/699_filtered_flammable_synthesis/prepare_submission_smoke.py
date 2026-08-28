from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

CATEGORIES = ("БАД", "Легковоспламеняющиеся")
OUTPUT_RE = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$", re.DOTALL)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_cache_path(cache: Path, row_id: str) -> Path:
    return cache / f"{hashlib.sha256(row_id.encode()).hexdigest()}.img"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepare(
    *,
    source_validation: Path,
    image_cache: Path,
    output_dir: Path,
    approved_s3_root: Path = Path("/work/s3"),
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite submission smoke input")
    rows = read_jsonl(source_validation)
    if any("label" in row for row in rows):
        raise ValueError("submission smoke source must be label-free")
    selected = []
    for category in CATEGORIES:
        local = sorted(
            (row for row in rows if row.get("category") == category),
            key=lambda row: (int(row["global_index"]), str(row["id"])),
        )[:4]
        if len(local) != 4:
            raise ValueError(f"insufficient smoke rows for {category}")
        selected.extend(local)
    output_dir.mkdir(parents=True)
    images = output_dir / "images"
    images.mkdir()
    data_path = output_dir / "data.csv"
    with data_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["id", "name", "description", "category"]
        )
        writer.writeheader()
        for row in selected:
            row_id = str(row["id"])
            writer.writerow(
                {
                    "id": row_id,
                    "name": str(row["name"]),
                    "description": str(row["description"]),
                    "category": str(row["category"]),
                }
            )
            cached = image_cache_path(image_cache, row_id)
            if not cached.is_symlink():
                raise ValueError("smoke image cache entry must be an immutable symlink")
            target = cached.resolve(strict=True)
            if (
                not target.is_file()
                or not target.is_relative_to(approved_s3_root.resolve(strict=True))
            ):
                raise ValueError("smoke image must resolve inside the approved S3 mount")
            destination = images / row_id
            destination.mkdir()
            os.symlink(target, destination / "0.jpg")
    report: dict[str, Any] = {
        "schema": "exp699_submission_smoke_input_v1",
        "experiment_id": "699",
        "rows": len(selected),
        "categories": {
            category: sum(row["category"] == category for row in selected)
            for category in CATEGORIES
        },
        "source_validation_sha256": sha256_file(source_validation),
        "selected_rows_sha256": canonical_sha256(
            [
                {
                    "id": str(row["id"]),
                    "global_index": int(row["global_index"]),
                    "category": str(row["category"]),
                }
                for row in selected
            ]
        ),
        "data_sha256": sha256_file(data_path),
        "image_count": len(selected),
        "labels_read": 0,
        "sealed_rows_used": 0,
        "public_rows_used": 0,
        "decision": "GO_PACKAGE_RUNTIME_SMOKE",
    }
    report["self_sha256"] = canonical_sha256(report)
    (output_dir / "smoke_input_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def verify_output(input_dir: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(
        (input_dir / "smoke_input_manifest.json").read_text(encoding="utf-8")
    )
    payload = dict(manifest)
    declared = payload.pop("self_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError("smoke input manifest self-hash mismatch")
    with (input_dir / "data.csv").open(encoding="utf-8", newline="") as stream:
        inputs = list(csv.DictReader(stream))
    with output_path.open(encoding="utf-8", newline="") as stream:
        outputs = list(csv.DictReader(stream))
    if [row["id"] for row in outputs] != [row["id"] for row in inputs]:
        raise ValueError("submission smoke IDs/order mismatch")
    if any(set(row) != {"id", "result"} for row in outputs):
        raise ValueError("submission smoke output columns mismatch")
    if any(OUTPUT_RE.fullmatch(str(row["result"])) is None for row in outputs):
        raise ValueError("submission smoke result schema mismatch")
    return {
        "schema": "exp699_submission_smoke_acceptance_v1",
        "experiment_id": "699",
        "input_manifest_self_sha256": declared,
        "rows": len(outputs),
        "output_sha256": sha256_file(output_path),
        "labels_read": 0,
        "sealed_rows_used": 0,
        "public_rows_used": 0,
        "decision": "ACCEPT_PACKAGE_RUNTIME_SMOKE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-validation", type=Path, required=True)
    prepare_parser.add_argument("--image-cache", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--approved-s3-root", type=Path, default=Path("/work/s3")
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--input-dir", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        prepare(
            source_validation=args.source_validation,
            image_cache=args.image_cache,
            output_dir=args.output_dir,
            approved_s3_root=args.approved_s3_root,
        )
        if args.command == "prepare"
        else verify_output(args.input_dir, args.output)
    )
    if "self_sha256" not in result:
        result["self_sha256"] = canonical_sha256(result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
