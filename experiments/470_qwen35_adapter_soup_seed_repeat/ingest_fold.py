from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
BASE = ROOT / "research/oof-cache-extracted/oof_scores.npz"
DESTINATION = EXPERIMENT / "artifacts/seed_31415"
MAX_OUTER_BYTES = 250 * 1024 * 1024
MAX_INNER_BYTES = 180 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(archive: zipfile.ZipFile, limit: int) -> list[zipfile.ZipInfo]:
    members = []
    total = 0
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(mode):
            raise ValueError(f"unsafe ZIP member: {info.filename}")
        total += info.file_size
        members.append(info)
    if total > limit:
        raise ValueError(f"ZIP expands above safe limit: {total}")
    bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    return members


def locate_inner(bundle: Path, temporary: Path, fold: int) -> Path:
    with zipfile.ZipFile(bundle) as archive:
        safe_members(archive, MAX_OUTER_BYTES)
        archive.extractall(temporary / "outer")
    matches = list((temporary / "outer").rglob(f"output_bundle_fold_{fold}.zip"))
    if len(matches) == 1:
        return matches[0]
    direct = list((temporary / "outer").rglob("soup_holdout_report.json"))
    if len(direct) == 1:
        return bundle
    raise ValueError(f"expected one inner fold bundle, found {len(matches)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=[0, 3], required=True)
    args = parser.parse_args()
    destination = DESTINATION / f"fold_{args.fold}"
    audit_path = EXPERIMENT / "analysis" / f"ingest_fold_{args.fold}.json"
    if destination.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite accepted fold {args.fold}")

    with tempfile.TemporaryDirectory(prefix="ingest_470_") as temp_name:
        temporary = Path(temp_name)
        inner = locate_inner(args.bundle, temporary, args.fold)
        inner_root = temporary / "inner"
        with zipfile.ZipFile(inner) as archive:
            safe_members(archive, MAX_INNER_BYTES)
            archive.extractall(inner_root)
        required = {
            "lora_holdout_predictions.csv",
            "lora_holdout_report.json",
            "adapter.zip",
            "soup_holdout_predictions.csv",
            "soup_holdout_report.json",
        }
        located: dict[str, Path] = {}
        for name in required:
            matches = list(inner_root.rglob(name))
            if len(matches) != 1:
                raise ValueError(f"expected one {name}, found {len(matches)}")
            located[name] = matches[0]

        base = np.load(BASE, allow_pickle=True)
        ids = base["ids"].astype(str)
        labels = base["labels"].astype(np.int8)
        categories = base["categories"].astype(str)
        folds = base["fold_ids"].astype(np.int8)
        pure_positions = np.flatnonzero(folds == args.fold)
        soup_positions = np.flatnonzero(
            (folds == args.fold) & (categories == "Легковоспламеняющиеся")
        )
        pure = pd.read_csv(located["lora_holdout_predictions.csv"], dtype={"id": str})
        soup = pd.read_csv(located["soup_holdout_predictions.csv"], dtype={"id": str})
        for name, frame, positions in (
            ("pure", pure, pure_positions), ("soup", soup, soup_positions)
        ):
            if not np.array_equal(frame.id.astype(str).to_numpy(), ids[positions]):
                raise ValueError(f"{name} id/order mismatch")
            if not np.array_equal(frame.label.astype(np.int8).to_numpy(), labels[positions]):
                raise ValueError(f"{name} label mismatch")
            if not np.array_equal(frame.fold.astype(np.int8).to_numpy(), folds[positions]):
                raise ValueError(f"{name} fold mismatch")
            if not np.array_equal(frame.category.astype(str).to_numpy(), categories[positions]):
                raise ValueError(f"{name} category mismatch")
            if not np.isfinite(frame.lora_score.astype(float).to_numpy()).all():
                raise ValueError(f"{name} contains non-finite scores")

        train_report = json.loads(located["lora_holdout_report.json"].read_text())
        soup_report = json.loads(located["soup_holdout_report.json"].read_text())
        if int(train_report.get("holdout_fold", -1)) != args.fold:
            raise ValueError("training report fold mismatch")
        if (
            soup_report.get("experiment_id") != "470"
            or int(soup_report.get("fold", -1)) != args.fold
            or int(soup_report.get("seed", -1)) != 31415
            or float(soup_report.get("alpha_specialist", -1)) != 0.5
        ):
            raise ValueError("soup report protocol mismatch")
        if soup_report.get("predictions_sha256") != sha256(located["soup_holdout_predictions.csv"]):
            raise ValueError("soup prediction checksum mismatch")
        with zipfile.ZipFile(located["adapter.zip"]) as adapter:
            safe_members(adapter, 80 * 1024 * 1024)

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        for name, source in located.items():
            shutil.copy2(source, destination / name)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "experiment_id": "470",
        "fold": args.fold,
        "bundle": str(args.bundle),
        "bundle_bytes": args.bundle.stat().st_size,
        "bundle_sha256": sha256(args.bundle),
        "pure_rows": len(pure),
        "soup_rows": len(soup),
        "accepted_files": {name: sha256(destination / name) for name in sorted(required)},
        "passed": True,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
