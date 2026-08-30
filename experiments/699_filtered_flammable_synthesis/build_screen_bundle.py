from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_file(archive: tarfile.TarFile, source: Path, destination: str) -> dict:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"bundle source must be a regular file: {source}")
    payload = source.read_bytes()
    info = tarfile.TarInfo(destination)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(payload))
    return {
        "path": destination,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_bundle(
    *, repo: Path, output: Path, manifest_path: Path
) -> dict:
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite bundle or manifest")
    members = [
        (
            repo / "experiments/699_filtered_flammable_synthesis/build_synth_runtime.py",
            "code/build_synth_runtime.py",
        ),
        (
            repo / "experiments/699_filtered_flammable_synthesis/train_synth_fold.py",
            "code/train_synth_fold.py",
        ),
        (
            repo / "research/peft-vendor-extracted/peft-0.20.0.zip",
            "peft-0.20.0.zip",
        ),
    ]
    for fold in (0, 3):
        for name in ("train.jsonl", "validation.jsonl", "runtime_audit.json"):
            members.append(
                (
                    repo
                    / "experiments/641_qwen35_4b_class_only_lora/.local/runtime"
                    / f"fold{fold}"
                    / name,
                    f"parent_runtime/fold{fold}/{name}",
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = []
    with (
        output.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for source, destination in members:
            inventory.append(add_file(archive, source, destination))
    manifest = {
        "schema_version": 1,
        "experiment_id": "699",
        "scope": "gpu_screen_folds03",
        "members": inventory,
        "member_count": len(inventory),
        "bundle_sha256": sha256(output),
        "bundle_size": output.stat().st_size,
    }
    manifest["self_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_bundle(
                repo=args.repo.resolve(),
                output=args.output.resolve(),
                manifest_path=args.manifest.resolve(),
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
