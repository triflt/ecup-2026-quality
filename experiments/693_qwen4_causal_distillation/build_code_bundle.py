from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_NAME = "QWEN4_CODE_BUNDLE_MANIFEST.json"
BUNDLE_PATHS = (
    "experiments/693_qwen4_causal_distillation",
    "experiments/694_qwen4_hardneg_curriculum",
    "experiments/695_qwen4_hard_anchored_listwise",
    "experiments/645_qwen_scale_2x3_gate/grid_contract.py",
    "experiments/645_qwen_scale_2x3_gate/train_lora.py",
    "experiments/645_qwen_scale_2x3_gate/frozen_spec.json",
    "experiments/641_qwen35_4b_class_only_lora/frozen_spec.json",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def normalized_info(source: tarfile.TarInfo) -> tarfile.TarInfo:
    info = tarfile.TarInfo(PurePosixPath(source.name).as_posix().removeprefix("./"))
    info.size = source.size
    info.mode = source.mode
    info.type = source.type
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build(repo: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite code bundle")
    if git(repo, "status", "--porcelain"):
        raise ValueError("immutable bundle requires a clean worktree")
    revision = git(repo, "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="qwen4_code_bundle_") as directory:
        source_tar = Path(directory) / "source.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                "-o",
                str(source_tar),
                "HEAD",
                *BUNDLE_PATHS,
            ],
            check=True,
        )
        members: list[tuple[tarfile.TarInfo, bytes | None]] = []
        files: dict[str, dict[str, Any]] = {}
        directories: set[str] = set()
        names: set[str] = set()
        with tarfile.open(source_tar) as source:
            for member in source.getmembers():
                info = normalized_info(member)
                path = PurePosixPath(info.name)
                if (
                    not info.name
                    or path.is_absolute()
                    or ".." in path.parts
                    or any(
                        part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
                        for part in path.parts
                    )
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"unsafe git archive member: {member.name}")
                if info.name in names:
                    raise ValueError("duplicate git archive member")
                names.add(info.name)
                payload: bytes | None = None
                if member.isfile():
                    extracted = source.extractfile(member)
                    if extracted is None:
                        raise ValueError("git archive member is unreadable")
                    payload = extracted.read()
                    files[info.name] = {
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                else:
                    directories.add(info.name.rstrip("/"))
                members.append((info, payload))
        manifest = {
            "schema_version": "qwen4_code_bundle_v1",
            "experiment_ids": ["693", "694", "695"],
            "git_revision": revision,
            "source_paths": list(BUNDLE_PATHS),
            "files": files,
            "directories": sorted(directories),
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_payload = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        manifest_info = tarfile.TarInfo(MANIFEST_NAME)
        manifest_info.size = len(manifest_payload)
        manifest_info.mode = 0o644
        manifest_info.mtime = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with (
            output.open("xb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as destination,
        ):
            for info, payload in members:
                destination.addfile(info, io.BytesIO(payload) if payload is not None else None)
            destination.addfile(manifest_info, io.BytesIO(manifest_payload))
    return {
        "git_revision": revision,
        "bundle_sha256": sha256_file(output),
        "manifest_sha256": manifest["manifest_sha256"],
        "bundle_size": output.stat().st_size,
        "source_paths": list(BUNDLE_PATHS),
        "files": len(files),
        "directories": len(directories),
    }


def verify(
    root: Path, *, expected_revision: str, archive: Path, expected_bundle_sha256: str
) -> dict[str, Any]:
    if sha256_file(archive) != expected_bundle_sha256:
        raise ValueError("code bundle archive SHA-256 mismatch")
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    manifest_sha256 = body.pop("manifest_sha256", None)
    if manifest_sha256 != canonical_sha256(body):
        raise ValueError("code bundle manifest self-hash mismatch")
    if (
        manifest.get("schema_version") != "qwen4_code_bundle_v1"
        or manifest.get("experiment_ids") != ["693", "694", "695"]
        or manifest.get("git_revision") != expected_revision
        or manifest.get("source_paths") != list(BUNDLE_PATHS)
    ):
        raise ValueError("code bundle manifest contract mismatch")
    expected_files = manifest.get("files")
    expected_directories = manifest.get("directories")
    if not isinstance(expected_files, dict) or not isinstance(expected_directories, list):
        raise TypeError("code bundle inventory is invalid")
    observed_files: dict[str, dict[str, Any]] = {}
    observed_directories: list[str] = []
    resolved = root.resolve()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(resolved):
            raise ValueError("code bundle contains an escaping member")
        if path.is_file() and relative != MANIFEST_NAME:
            observed_files[relative] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        elif path.is_dir():
            observed_directories.append(relative)
        elif not path.is_file():
            raise ValueError("code bundle contains a special member")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("code bundle extracted inventory mismatch")
    return {
        "schema_version": "qwen4_code_bundle_acceptance_v1",
        "git_revision": expected_revision,
        "bundle_sha256": expected_bundle_sha256,
        "manifest_sha256": manifest_sha256,
        "source_paths": list(BUNDLE_PATHS),
        "decision": "ACCEPT_CODE_BUNDLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--repo", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--expected-revision", required=True)
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.add_argument("--expected-bundle-sha256", required=True)
    args = parser.parse_args()
    result = (
        build(args.repo, args.output)
        if args.command == "build"
        else verify(
            args.root,
            expected_revision=args.expected_revision,
            archive=args.archive,
            expected_bundle_sha256=args.expected_bundle_sha256,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
