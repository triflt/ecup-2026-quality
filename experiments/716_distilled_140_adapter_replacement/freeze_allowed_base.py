from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

EXPECTED_TREE_SHA256 = "3a237b0aaaacbdb43ec8f40aafc55db180e3beb1447125b81c305d42d5d46cfe"
ALLOWED_MODELS = {
    "Qwen/Qwen3-VL-Embedding-2B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3.5-4B",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_files(root: Path) -> dict[str, Path]:
    files = {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if not files:
        raise ValueError("allowed base source directory is empty")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite frozen base")
    files = source_files(args.source_dir)
    hashes = {name: sha256(path) for name, path in files.items()}
    tree_sha256 = canonical_sha256(hashes)
    if tree_sha256 != EXPECTED_TREE_SHA256:
        raise ValueError("allowed base source tree checksum mismatch")
    required = {
        "run.py",
        "metadata.json",
        "adapter_qwen35/adapter_model.safetensors",
        "adapter_qwen3vl/adapter_model.safetensors",
    }
    if not required.issubset(files):
        raise ValueError("allowed base source lacks required runtime components")
    run_text = files["run.py"].read_text(encoding="utf-8")
    if any(model not in run_text for model in ALLOWED_MODELS):
        raise ValueError("allowed model mount is missing from base runtime")
    forbidden = ("Giga-Embeddings", "Qwen3.8-27B", "Qwen3.8")
    if any(value in run_text for value in forbidden):
        raise ValueError("base runtime references a forbidden model")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name, path in files.items():
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (path.stat().st_mode & 0o777) << 16
                archive.writestr(info, path.read_bytes())
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("frozen allowed base ZIP integrity failure")
    report = {
        "schema_version": "exp716_allowed_base_freeze_v1",
        "experiment_id": "716",
        "source_read_only": True,
        "excluded": ["**/__pycache__/**", "**/*.pyc"],
        "file_count": len(files),
        "source_tree_sha256": tree_sha256,
        "allowed_model_mounts": sorted(ALLOWED_MODELS),
        "forbidden_model_references_absent": True,
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "zip_integrity": "PASS",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
