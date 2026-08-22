from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DESTINATION = EXPERIMENT / "artifacts/oof"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
FLAMMABLE = "Легковоспламеняющиеся"
MAX_UNCOMPRESSED_BYTES = 300 * 1024 * 1024


def sha256(path: Path) -> str:
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


def expected_rows() -> list[dict[str, str]]:
    with FOLDS.open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["category"] == FLAMMABLE]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--alpha", type=float, choices=[0.25, 0.5, 0.75], required=True)
    args = parser.parse_args()
    slug = f"alpha_{int(args.alpha * 100):02d}"
    destination = DESTINATION / slug
    audit_path = EXPERIMENT / f"analysis/ingest_{slug}.json"
    if destination.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite existing alpha artifact: {slug}")

    with zipfile.ZipFile(args.bundle) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure: {bad}")
        total = 0
        names = set()
        for info in archive.infolist():
            name = safe_name(info.filename)
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise ValueError(f"symbolic link is not allowed: {info.filename}")
            total += info.file_size
            names.add(name.name)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError(f"ZIP expands above safe limit: {total}")
        required = {"soup_oof_predictions.csv", "soup_oof_report.json"}
        if not required <= names:
            raise ValueError(f"required files absent: {sorted(required - names)}")
        with tempfile.TemporaryDirectory(prefix="ingest_430_", dir=EXPERIMENT / "artifacts") as temp:
            staging = Path(temp)
            for info in archive.infolist():
                if safe_name(info.filename).name in required:
                    archive.extract(info, staging)
            located = {}
            for filename in required:
                matches = list(staging.rglob(filename))
                if len(matches) != 1:
                    raise ValueError(f"expected one {filename}, found {len(matches)}")
                located[filename] = matches[0]
            report = json.loads(located["soup_oof_report.json"].read_text())
            if report.get("experiment_id") != "430" or float(report.get("alpha_260")) != args.alpha:
                raise ValueError("experiment/alpha report mismatch")
            if int(report.get("download_failures", -1)) != 0:
                raise ValueError("report contains image download failures")
            if len(report.get("folds", [])) != 5 or len(report.get("merge_reports", [])) != 5:
                raise ValueError("report does not contain five folds and merges")
            predictions = located["soup_oof_predictions.csv"]
            with predictions.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            expected = expected_rows()
            for field in ("id", "category", "label", "fold"):
                if [str(row[field]) for row in rows] != [str(row[field]) for row in expected]:
                    raise ValueError(f"prediction {field} mismatch")
            values = [float(row["lora_score"]) for row in rows]
            if any(value != value or abs(value) == float("inf") for value in values):
                raise ValueError("prediction score is not finite")
            if report.get("predictions_sha256") != sha256(predictions):
                raise ValueError("prediction checksum mismatch with report")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.mkdir()
            shutil.copy2(predictions, destination / predictions.name)
            shutil.copy2(located["soup_oof_report.json"], destination / "soup_oof_report.json")

    audit = {
        "experiment_id": "430",
        "alpha_260": args.alpha,
        "bundle": str(args.bundle),
        "bundle_bytes": args.bundle.stat().st_size,
        "bundle_sha256": sha256(args.bundle),
        "prediction_rows": len(rows),
        "predictions_sha256": sha256(destination / "soup_oof_predictions.csv"),
        "report_sha256": sha256(destination / "soup_oof_report.json"),
        "passed": True,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
