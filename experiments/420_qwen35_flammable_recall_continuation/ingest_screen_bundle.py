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
DEFAULT_DESTINATION = EXPERIMENT / "artifacts/screen_seed42"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
MANIFEST = EXPERIMENT / "analysis/training_manifest.json"
PARENT = ROOT / "experiments/260_bad_family_diverse_positives/artifacts/seed_42_diverse_positives"
FLAMMABLE = "Легковоспламеняющиеся"
MAX_UNCOMPRESSED_BYTES = 300 * 1024 * 1024


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


def validate_archive(archive: zipfile.ZipFile) -> None:
    bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"CRC failure in ZIP member: {bad}")
    total = 0
    for info in archive.infolist():
        safe_name(info.filename)
        file_type = (info.external_attr >> 16) & 0o170000
        if file_type == 0o120000:
            raise ValueError(f"symbolic link is not allowed in ZIP: {info.filename}")
        total += info.file_size
    if total > MAX_UNCOMPRESSED_BYTES:
        raise ValueError(f"ZIP expands to {total} bytes, above safe limit")


def stages(archive: zipfile.ZipFile) -> set[str]:
    result = set()
    for info in archive.infolist():
        parts = safe_name(info.filename).parts
        if parts and parts[0].startswith("fold_"):
            result.add(parts[0])
    return result


def open_payload(path: Path) -> tuple[zipfile.ZipFile, io.BytesIO | None, str]:
    outer = zipfile.ZipFile(path)
    validate_archive(outer)
    if stages(outer):
        return outer, None, "outer"
    inner = [
        info
        for info in outer.infolist()
        if PurePosixPath(info.filename).name.startswith("output_bundle_")
        and info.filename.endswith(".zip")
    ]
    if len(inner) != 1:
        outer.close()
        raise ValueError(f"expected direct stage or one output bundle, found {len(inner)}")
    payload_location = inner[0].filename
    buffer = io.BytesIO(outer.read(inner[0]))
    outer.close()
    archive = zipfile.ZipFile(buffer)
    validate_archive(archive)
    return archive, buffer, payload_location


def expected_rows(fold: int) -> list[dict[str, str]]:
    with FOLDS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row
        for row in rows
        if int(row["fold"]) == fold and row["category"] == FLAMMABLE
    ]


def validate_stage(root: Path, stage: str) -> dict[str, object]:
    fold = int(stage.removeprefix("fold_"))
    stage_root = root / stage
    adapter = stage_root / "adapter/adapter_model.safetensors"
    config = stage_root / "adapter/adapter_config.json"
    report_path = stage_root / "lora_holdout_report.json"
    predictions = stage_root / "lora_holdout_predictions.csv"
    if not adapter.is_file() or adapter.stat().st_size < 1_000_000:
        raise ValueError(f"missing or implausibly small adapter for {stage}")
    json.loads(config.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("experiment_id") != "420" or report.get("stage") != stage:
        raise ValueError(f"experiment/stage mismatch in {stage} report")
    if int(report.get("holdout_fold", -1)) != fold:
        raise ValueError(f"holdout fold mismatch in {stage} report")
    if int(report.get("download_failures", -1)) != 0:
        raise ValueError(f"download failures recorded for {stage}")
    manifest_sha = sha256_path(MANIFEST)
    if report.get("training_manifest_sha256") != manifest_sha:
        raise ValueError(f"training manifest hash mismatch for {stage}")
    parent_config = PARENT / stage / "adapter/adapter_config.json"
    parent_config_sha = sha256_path(parent_config)
    if report.get("parent_adapter_config_sha256") != parent_config_sha:
        raise ValueError(f"parent adapter config hash mismatch for {stage}")

    expected = expected_rows(fold)
    with predictions.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["id"] for row in rows] != [row["id"] for row in expected]:
        raise ValueError(f"prediction id/order mismatch for {stage}")
    if any(row["category"] != FLAMMABLE for row in rows):
        raise ValueError(f"unexpected category in {stage} predictions")
    if any(int(row["fold"]) != fold for row in rows):
        raise ValueError(f"unexpected fold in {stage} predictions")
    if [int(row["label"]) for row in rows] != [int(row["label"]) for row in expected]:
        raise ValueError(f"prediction label mismatch for {stage}")
    if len(rows) != int(report.get("validation_rows", -1)):
        raise ValueError(f"validation row count mismatch for {stage}")
    scores = [float(row["lora_score"]) for row in rows]
    if any(score != score or abs(score) == float("inf") for score in scores):
        raise ValueError(f"non-finite prediction score for {stage}")
    return {
        "stage": stage,
        "adapter_bytes": adapter.stat().st_size,
        "adapter_sha256": sha256_path(adapter),
        "adapter_config_sha256": sha256_path(config),
        "prediction_rows": len(rows),
        "predictions_sha256": sha256_path(predictions),
        "report_sha256": sha256_path(report_path),
        "standalone_flammable_f1": float(report["standalone_flammable_f1"]),
        "training_runtime_minutes": float(report["training_runtime_minutes"]),
        "total_runtime_minutes": float(report["total_runtime_minutes"]),
        "training_manifest_sha256": manifest_sha,
        "parent_adapter_config_sha256": parent_config_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage", choices=["fold_0", "fold_3"], required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    destination = args.destination / args.stage
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing stage: {destination}")
    audit_path = args.audit or EXPERIMENT / f"analysis/ingest_screen_{args.stage}.json"
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {audit_path}")

    archive, buffer, payload_location = open_payload(args.bundle)
    actual = stages(archive)
    if actual != {args.stage}:
        archive.close()
        raise ValueError(f"stage mismatch: archive={sorted(actual)} expected={[args.stage]}")

    args.destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ingest_420_", dir=args.destination.parent) as temp:
        staging = Path(temp)
        for info in archive.infolist():
            name = safe_name(info.filename)
            if name.parts and name.parts[0] == args.stage:
                archive.extract(info, staging)
        archive.close()
        stage_audit = validate_stage(staging, args.stage)
        shutil.move(str(staging / args.stage), str(destination))

    audit = {
        "bundle": str(args.bundle),
        "bundle_bytes": args.bundle.stat().st_size,
        "bundle_sha256": sha256_path(args.bundle),
        "payload_location": payload_location,
        "stage": stage_audit,
        "passed": True,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
