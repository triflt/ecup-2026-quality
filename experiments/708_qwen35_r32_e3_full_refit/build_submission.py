from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ADAPTER_FILES = {"README.md", "adapter_config.json", "adapter_model.safetensors"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files(root: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinked package member: {path.relative_to(root)}")
        if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store":
            result[path.relative_to(root).as_posix()] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists() or args.archive.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite exp708 package output")
    source_files = files(args.source)
    required = {"run.py", "metadata.json", "adapter_qwen35/adapter_model.safetensors"}
    if not required.issubset(source_files):
        raise ValueError("solution140 source is incomplete")
    adapter_files = files(args.adapter)
    if set(adapter_files) != ADAPTER_FILES:
        raise ValueError(f"unexpected adapter members: {sorted(adapter_files)}")
    config = json.loads(adapter_files["adapter_config.json"].read_text())
    if config.get("r") != 32 or config.get("lora_alpha") != 64 or not config.get("use_rslora"):
        raise ValueError("adapter is not the accepted rsLoRA r32/alpha64 recipe")

    shutil.copytree(args.source, args.destination, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    shutil.rmtree(args.destination / "adapter_qwen35")
    shutil.copytree(args.adapter, args.destination / "adapter_qwen35")
    candidate_files = files(args.destination)
    changed = []
    unchanged = {}
    for relative, source_path in source_files.items():
        if relative.startswith("adapter_qwen35/"):
            continue
        if sha256(candidate_files[relative]) != sha256(source_path):
            changed.append(relative)
        else:
            unchanged[relative] = sha256(source_path)
    if changed:
        raise ValueError(f"non-adapter solution140 members changed: {changed}")

    manifest = {
        "schema": "exp708_solution140_qwen35_r32_epoch3_full_refit_v1",
        "parent": "solution140",
        "changed_factor": "adapter_qwen35_rsLoRA_r16_epoch1_to_r32_epoch3",
        "adapter": {name: sha256(path) for name, path in adapter_files.items()},
        "unchanged_members": len(unchanged),
        "qwen3vl_unchanged": True,
        "fusion_weights_unchanged": True,
        "thresholds_unchanged": True,
        "annotator_prior_unchanged": True,
        "public_used": False,
    }
    manifest_path = args.destination / "exp708_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(args.archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for relative, path in files(args.destination).items():
            output.write(path, relative)
    manifest.update({"archive_sha256": sha256(args.archive), "archive_size": args.archive.stat().st_size})
    args.report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
