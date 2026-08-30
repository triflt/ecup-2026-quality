from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import build_dataset

EXPECTED_SHARDS = 32
EXPECTED_MEMBERS = {"report.json", "spotting.jsonl"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_shards(archive_dir: Path, output_dir: Path) -> list[Path]:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite extracted shards")
    output_dir.mkdir(parents=True)
    shard_dirs: list[Path] = []
    for shard in range(EXPECTED_SHARDS):
        archive_path = archive_dir / f"paddleocr_full_s{shard:02d}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"shard {shard} ZIP CRC failure: {bad}")
            members = set(archive.namelist())
            if members != EXPECTED_MEMBERS:
                raise ValueError(f"shard {shard} member set mismatch")
            if any(
                PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                for name in members
            ):
                raise ValueError(f"shard {shard} contains an unsafe path")
            shard_dir = output_dir / f"shard{shard:02d}"
            shard_dir.mkdir()
            archive.extractall(shard_dir)
        shard_dirs.append(shard_dir)
    return shard_dirs


def download_source_archives(source_manifest: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite downloaded source archives")
    rows = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != EXPECTED_SHARDS:
        raise ValueError("source URL manifest must contain exactly 32 rows")
    by_shard: dict[int, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"shard", "url", "archive_sha256"}:
            raise ValueError("source URL manifest row schema mismatch")
        shard = row["shard"]
        url = row["url"]
        expected_sha = row["archive_sha256"]
        if type(shard) is not int or not 0 <= shard < EXPECTED_SHARDS or shard in by_shard:
            raise ValueError("invalid or duplicate source shard")
        if not isinstance(url, str) or not url.startswith("https://") or any(
            char.isspace() for char in url
        ):
            raise ValueError("source archive URL must be HTTPS")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(
            char not in "0123456789abcdef" for char in expected_sha
        ):
            raise ValueError("invalid source archive SHA-256")
        by_shard[shard] = {"url": url, "archive_sha256": expected_sha}
    if set(by_shard) != set(range(EXPECTED_SHARDS)):
        raise ValueError("source URL manifest has incomplete shard indices")
    output_dir.mkdir(parents=True)

    def download(shard: int) -> None:
        destination = output_dir / f"paddleocr_full_s{shard:02d}.zip"
        request = urllib.request.Request(
            by_shard[shard]["url"], headers={"User-Agent": "Mozilla/5.0"}
        )
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=180) as response, destination.open(
            "wb"
        ) as output:
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != by_shard[shard]["archive_sha256"]:
            raise ValueError(f"downloaded source archive checksum mismatch for shard {shard}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, shard) for shard in range(EXPECTED_SHARDS)]
        for future in as_completed(futures):
            future.result()
    return output_dir


def package_dataset(dataset_dir: Path, audit: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "competition_train_paddleocr_vl16_spotting_v1.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dataset_dir.iterdir()):
            if path.is_file():
                archive.write(path, str(Path("dataset") / path.name))
        archive.writestr(
            "acceptance_audit.json",
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"packaged dataset ZIP CRC failure: {bad}")
        members = sorted(archive.namelist())
    delivery = {
        "schema_version": 1,
        "experiment_id": "634",
        "dataset_version": build_dataset.DATASET_VERSION,
        "dataset_sha256": audit["dataset_sha256"],
        "bundle_sha256": audit["bundle_sha256"],
        "archive_sha256": sha256_file(archive_path),
        "archive_members": len(members),
        "records": audit["records"],
        "items": audit["items"],
        "decision": "READY_FOR_LOCAL_REVERIFY_AND_REGISTRATION",
    }
    (output_dir / "delivery.json").write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return delivery


def run(
    manifest: Path,
    archive_dir: Path | None,
    source_url_manifest: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    if (archive_dir is None) == (source_url_manifest is None):
        raise ValueError("choose exactly one OCR archive source")
    with tempfile.TemporaryDirectory(prefix="ocr-dataset-634-") as temporary_raw:
        temporary = Path(temporary_raw)
        resolved_archive_dir = (
            archive_dir
            if archive_dir is not None
            else download_source_archives(source_url_manifest, temporary / "source_archives")
        )
        shard_dirs = extract_shards(resolved_archive_dir, temporary / "shards")
        dataset_dir = temporary / build_dataset.DATASET_VERSION
        built = build_dataset.build_dataset(
            manifest_path=manifest,
            shard_dirs=shard_dirs,
            output_dir=dataset_dir,
        )
        verified = build_dataset.verify_bundle(dataset_dir)
        if built != verified:
            raise RuntimeError("build and independent verify reports differ")
        audit = {
            **verified,
            "source_archives": {
                f"paddleocr_full_s{shard:02d}.zip": sha256_file(
                    resolved_archive_dir / f"paddleocr_full_s{shard:02d}.zip"
                )
                for shard in range(EXPECTED_SHARDS)
            },
            "source_archive_count": EXPECTED_SHARDS,
            "all_source_zip_crc_passed": True,
            "verify_only_passed": True,
            "decision": "ACCEPT_IMMUTABLE_DATASET",
        }
        result = package_dataset(dataset_dir, audit, output_dir)
        shutil.rmtree(temporary / "shards")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build experiment 634 in an isolated runtime.")
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive-dir", type=Path)
    source.add_argument("--source-url-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.manifest, args.archive_dir, args.source_url_manifest, args.output_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
