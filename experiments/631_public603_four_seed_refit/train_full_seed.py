from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

PARENT_SHA256 = "c30e690ad260af72fcc625c8d3e6d9ab9c5a096d8443d6d9f5f7adbcaa52123c"
DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
OOF_SHA256 = "d78bb7df897e244e030fa388e1b9c770d15434cc33aa1f56b238c7d057646116"
MANIFEST_SHA256 = "d6193215ce2d6145440bd484ea225e77fe10784c4c2c06b7c6b0b85dc8efc7d9"
ALLOWED_SEEDS = (161803, 271828)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"input checksum mismatch for {path.name}: {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-data refit for one frozen experiment-603 seed.")
    parser.add_argument("--seed", required=True, type=int, choices=ALLOWED_SEEDS)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--data-parts", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--vendor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path, expected in (
        (args.parent, PARENT_SHA256),
        (args.oof, OOF_SHA256),
        (args.manifest, MANIFEST_SHA256),
    ):
        require_sha(path, expected)
    parts = sorted(args.data_parts.glob("data.csv.gz.part-*"))
    if not parts:
        raise ValueError("no split data archive parts found")
    data_payload = gzip.decompress(b"".join(path.read_bytes() for path in parts))
    if hashlib.sha256(data_payload).hexdigest() != DATA_SHA256:
        raise ValueError("reconstructed data checksum mismatch")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "SEED": str(args.seed),
            "FULL_TRAIN": "1",
            "TRAINING_MODE": "hard",
            "MODEL_CLASS": "multimodal",
            "USE_CHAT_BATCH": "1",
            "DESCRIPTION_LIMIT": "1800",
            "ECUP_DATA": str((args.data_parts / "absent-data.csv").resolve()),
            "ECUP_DATA_PARTS": str(args.data_parts.resolve()),
            "ECUP_OOF": str(args.oof.resolve()),
            "ECUP_MANIFEST": str(args.manifest.resolve()),
            "ECUP_IMAGES": str(args.images.resolve()),
            "ECUP_MODEL_ROOT": str(args.model_root.resolve()),
            "ECUP_VENDOR": str(args.vendor.resolve()),
            "ECUP_OUTPUT_DIR": str(output),
        }
    )
    os.environ.clear()
    os.environ.update(environment)
    spec = importlib.util.spec_from_file_location("_exp631_frozen_parent", args.parent)
    if spec is None or spec.loader is None:
        raise ImportError("cannot import frozen parent trainer")
    parent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parent
    spec.loader.exec_module(parent)
    expected = {
        "SEED": args.seed,
        "FULL_TRAIN": True,
        "TRAINING_MODE": "hard",
        "MODEL_CLASS": "multimodal",
        "USE_CHAT_BATCH": True,
        "DESCRIPTION_LIMIT": 1800,
        "BATCH_SIZE": 4,
        "GRAD_ACCUM": 4,
        "EPOCHS": 1,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(parent, key, None)}
        for key, value in expected.items()
        if getattr(parent, key, None) != value
    }
    if mismatches:
        raise ValueError(f"frozen full-data recipe mismatch: {mismatches}")
    parent.main()
    report_path = output / "full_train_report.json"
    adapter_path = output / "adapter.zip"
    if not report_path.is_file() or not adapter_path.is_file():
        raise ValueError("full-data output is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("full_train") is not True or report.get("download_failures") != 0:
        raise ValueError("full-data report failed its integrity gate")
    contract = {
        "experiment_id": 631,
        "parent_experiment": 603,
        "seed": args.seed,
        "full_train": True,
        "training_mode": "hard",
        "adapter_sha256": sha256_file(adapter_path),
        "report_sha256": sha256_file(report_path),
        "train_records": report.get("train_records"),
        "train_unique": report.get("train_unique"),
        "optimizer_updates": report.get("optimizer_updates"),
        "runtime_minutes": report.get("runtime_minutes"),
    }
    (output / "full_seed_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
