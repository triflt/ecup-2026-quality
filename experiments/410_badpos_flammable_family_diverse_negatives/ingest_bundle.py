from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_DESTINATION = EXPERIMENT / "artifacts/seed_42_negdiv"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
MAX_UNCOMPRESSED_BYTES = 600 * 1024 * 1024


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name: str) -> PurePosixPath:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"unsafe ZIP member: {name}")
    return value


def validate_archive_members(archive: zipfile.ZipFile) -> None:
    total = 0
    for info in archive.infolist():
        safe_name(info.filename)
        file_type = (info.external_attr >> 16) & 0o170000
        if file_type == 0o120000:
            raise ValueError(f"symbolic link is not allowed in ZIP: {info.filename}")
        total += info.file_size
    if total > MAX_UNCOMPRESSED_BYTES:
        raise ValueError(f"ZIP expands to {total} bytes, above safe limit")


def stage_names(archive: zipfile.ZipFile) -> set[str]:
    values = set()
    for info in archive.infolist():
        parts = safe_name(info.filename).parts
        if parts and (parts[0] == "full" or parts[0].startswith("fold_")):
            values.add(parts[0])
    return values


def open_payload(path: Path) -> tuple[zipfile.ZipFile, io.BytesIO | None, str]:
    outer = zipfile.ZipFile(path)
    if outer.testzip() is not None:
        raise ValueError(f"CRC failure in outer ZIP: {outer.testzip()}")
    validate_archive_members(outer)
    if stage_names(outer):
        return outer, None, "outer"
    inner = [
        info
        for info in outer.infolist()
        if PurePosixPath(info.filename).name.startswith("output_bundle_")
        and info.filename.endswith(".zip")
    ]
    if len(inner) != 1:
        outer.close()
        raise ValueError(
            f"expected direct stages or one output_bundle ZIP, found {len(inner)}"
        )
    buffer = io.BytesIO(outer.read(inner[0]))
    outer.close()
    archive = zipfile.ZipFile(buffer)
    bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"CRC failure in inner ZIP: {bad}")
    validate_archive_members(archive)
    return archive, buffer, inner[0].filename


def expected_fold_rows() -> dict[str, int]:
    with FOLDS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        f"fold_{fold}": sum(int(row["fold"]) == fold for row in rows)
        for fold in range(5)
    }


def validate_stage(root: Path, stage: str, fold_rows: dict[str, int]) -> dict:
    stage_root = root / stage
    model = stage_root / "adapter/adapter_model.safetensors"
    config = stage_root / "adapter/adapter_config.json"
    if not model.is_file() or model.stat().st_size < 1_000_000:
        raise ValueError(f"missing or implausibly small adapter for {stage}")
    json.loads(config.read_text(encoding="utf-8"))
    result = {
        "stage": stage,
        "adapter_bytes": model.stat().st_size,
        "adapter_sha256": sha256_path(model),
    }
    if stage == "full":
        report = stage_root / "full_train_report.json"
        payload = json.loads(report.read_text(encoding="utf-8"))
        if not payload.get("full_train"):
            raise ValueError("full stage report is not marked full_train")
        result["train_records"] = int(payload["train_records"])
    else:
        report = stage_root / "lora_holdout_report.json"
        predictions = stage_root / "lora_holdout_predictions.csv"
        payload = json.loads(report.read_text(encoding="utf-8"))
        fold = int(stage.removeprefix("fold_"))
        if int(payload["holdout_fold"]) != fold:
            raise ValueError(f"holdout fold mismatch for {stage}")
        with predictions.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != fold_rows[stage]:
            raise ValueError(
                f"prediction row mismatch for {stage}: {len(rows)} != {fold_rows[stage]}"
            )
        if any(int(row["fold"]) != fold for row in rows):
            raise ValueError(f"prediction fold values mismatch for {stage}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate prediction ids for {stage}")
        result["prediction_rows"] = len(rows)
        result["predictions_sha256"] = sha256_path(predictions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()

    expected = set(args.stages)
    invalid = expected - {"fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "full"}
    if invalid:
        raise ValueError(f"invalid expected stages: {sorted(invalid)}")
    for stage in expected:
        if (args.destination / stage).exists():
            raise FileExistsError(f"refusing to overwrite existing stage: {stage}")
    audit_path = args.audit or EXPERIMENT / f"analysis/ingest_{args.suffix}.json"
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {audit_path}")

    archive, buffer, payload_location = open_payload(args.bundle)
    actual = stage_names(archive)
    if actual != expected:
        archive.close()
        raise ValueError(f"stage mismatch: archive={sorted(actual)} expected={sorted(expected)}")

    args.destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"ingest_{args.suffix}_", dir=args.destination.parent
    ) as temporary:
        staging = Path(temporary)
        for info in archive.infolist():
            name = safe_name(info.filename)
            if not name.parts or name.parts[0] not in expected:
                continue
            archive.extract(info, staging)
        archive.close()
        stage_audits = [
            validate_stage(staging, stage, expected_fold_rows())
            for stage in sorted(expected)
        ]
        for stage in sorted(expected):
            shutil.move(str(staging / stage), str(args.destination / stage))

    audit = {
        "bundle": str(args.bundle),
        "bundle_bytes": args.bundle.stat().st_size,
        "bundle_sha256": sha256_path(args.bundle),
        "payload_location": payload_location,
        "expected_stages": sorted(expected),
        "stages": stage_audits,
        "passed": True,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
