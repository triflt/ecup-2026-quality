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
MAX_BYTES = 250 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(archive: zipfile.ZipFile) -> None:
    total = 0
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(mode):
            raise ValueError(f"unsafe ZIP member: {info.filename}")
        total += info.file_size
    if total > MAX_BYTES:
        raise ValueError(f"ZIP expands above safe limit: {total}")
    bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")


def locate_inner(bundle: Path, temporary: Path, fold: int) -> Path:
    with zipfile.ZipFile(bundle) as archive:
        safe_members(archive)
        archive.extractall(temporary / "outer")
    matches = list((temporary / "outer").rglob(f"output_bundle_fold_{fold}.zip"))
    if len(matches) == 1:
        return matches[0]
    direct = list((temporary / "outer").rglob("fisher_holdout_report.json"))
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

    with tempfile.TemporaryDirectory(prefix="ingest_480_") as temp_name:
        temporary = Path(temp_name)
        inner = locate_inner(args.bundle, temporary, args.fold)
        root = temporary / "inner"
        with zipfile.ZipFile(inner) as archive:
            safe_members(archive)
            archive.extractall(root)
        required = {
            "fisher_holdout_predictions.csv",
            "fisher_holdout_report.json",
            "fisher_block_alphas.json",
        }
        located = {}
        for name in required:
            matches = list(root.rglob(name))
            if len(matches) != 1:
                raise ValueError(f"expected one {name}, found {len(matches)}")
            located[name] = matches[0]

        base = np.load(BASE, allow_pickle=True)
        ids = base["ids"].astype(str)
        labels = base["labels"].astype(np.int8)
        categories = base["categories"].astype(str)
        folds = base["fold_ids"].astype(np.int8)
        positions = np.flatnonzero((folds == args.fold) & (categories == "Легковоспламеняющиеся"))
        frame = pd.read_csv(located["fisher_holdout_predictions.csv"], dtype={"id": str})
        if not np.array_equal(frame.id.astype(str).to_numpy(), ids[positions]):
            raise ValueError("prediction id/order mismatch")
        if not np.array_equal(frame.label.astype(np.int8).to_numpy(), labels[positions]):
            raise ValueError("prediction label mismatch")
        if not np.array_equal(frame.fold.astype(np.int8).to_numpy(), folds[positions]):
            raise ValueError("prediction fold mismatch")
        if not np.isfinite(frame.lora_score.astype(float).to_numpy()).all():
            raise ValueError("non-finite Fisher scores")
        report = json.loads(located["fisher_holdout_report.json"].read_text())
        fisher = json.loads(located["fisher_block_alphas.json"].read_text())
        if (
            report.get("experiment_id") != "480"
            or int(report.get("fold", -1)) != args.fold
            or int(report.get("seed", -1)) != 31415
            or report.get("predictions_sha256") != sha256(located["fisher_holdout_predictions.csv"])
        ):
            raise ValueError("prediction report protocol mismatch")
        if (
            fisher.get("experiment_id") != "480"
            or int(fisher.get("fold", -1)) != args.fold
            or int(fisher.get("seed", -1)) != 31415
            or fisher.get("selection", {}).get("uses_holdout_labels") is not False
        ):
            raise ValueError("Fisher report protocol mismatch")
        alphas = np.asarray(list(fisher.get("block_alphas", {}).values()), dtype=float)
        if (
            len(alphas) == 0
            or not np.isfinite(alphas).all()
            or (alphas < 0.1).any()
            or (alphas > 0.9).any()
        ):
            raise ValueError("invalid Fisher block coefficients")

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        for name, source in located.items():
            shutil.copy2(source, destination / name)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "experiment_id": "480",
        "fold": args.fold,
        "bundle": str(args.bundle),
        "bundle_bytes": args.bundle.stat().st_size,
        "bundle_sha256": sha256(args.bundle),
        "prediction_rows": len(frame),
        "blocks": len(alphas),
        "accepted_files": {name: sha256(destination / name) for name in sorted(required)},
        "passed": True,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
