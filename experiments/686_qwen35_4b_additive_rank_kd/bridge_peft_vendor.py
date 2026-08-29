from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from build_pair_runtime import canonical_sha256

EXPERIMENT_ID = "685"
EXPECTED_MEMBER = "research/peft-vendor-extracted/peft-0.20.0.zip"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unsafe_parts(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        path.is_absolute()
        or ".." in path.parts
        or any(
            part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
            for part in path.parts
        )
    )


def bridge(source: Path, output: Path, *, expected_source_sha256: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite PEFT bridge output")
    if sha256_file(source) != expected_source_sha256:
        raise ValueError("PEFT source-bundle SHA mismatch")
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        names = [PurePosixPath(member.name).as_posix().removeprefix("./") for member in members]
        if len(names) != len(set(names)):
            raise ValueError("PEFT source bundle contains duplicate members")
        if any(unsafe_parts(name) for name in names):
            raise ValueError("PEFT source bundle contains unsafe transport metadata")
        matching = [
            member
            for member, name in zip(members, names, strict=True)
            if name == EXPECTED_MEMBER
        ]
        if len(matching) != 1 or not matching[0].isfile():
            raise ValueError("PEFT source bundle lacks the exact frozen vendor member")
        extracted = archive.extractfile(matching[0])
        if extracted is None:
            raise ValueError("PEFT source member is not readable")
        payload = extracted.read()
    if not payload:
        raise ValueError("PEFT vendor ZIP is empty")

    output.mkdir(parents=True)
    vendor = output / "peft-0.20.0.zip"
    vendor.write_bytes(payload)
    with zipfile.ZipFile(vendor) as archive:
        if archive.testzip() is not None:
            raise ValueError("PEFT vendor ZIP is corrupt")
        names = archive.namelist()
        normalized = [PurePosixPath(name).as_posix() for name in names]
        if len(normalized) != len(set(normalized)):
            raise ValueError("PEFT vendor ZIP contains duplicate members")
        if any(unsafe_parts(name) for name in normalized):
            raise ValueError("PEFT vendor ZIP contains unsafe transport metadata")
        required_members = {
            "peft/__init__.py",
            "peft/config.py",
            "peft/peft_model.py",
            "peft/tuners/lora/config.py",
            "peft/tuners/lora/model.py",
            "peft/utils/peft_types.py",
            "peft-0.20.0.dist-info/METADATA",
        }
        if not required_members.issubset(normalized):
            raise ValueError("PEFT vendor ZIP lacks required runtime modules")
        metadata = archive.read("peft-0.20.0.dist-info/METADATA").decode(
            "utf-8", errors="strict"
        )
        if "\nVersion: 0.20.0\n" not in f"\n{metadata.rstrip()}\n":
            raise ValueError("PEFT distribution metadata version mismatch")
    probe = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import json,pathlib,sys;"
                "sys.path.insert(0,sys.argv[1]);"
                "import peft;"
                "from peft import LoraConfig,PeftModel,TaskType,get_peft_model;"
                "print(json.dumps({'version':peft.__version__,'file':peft.__file__,"
                "'symbols':[x.__name__ if hasattr(x,'__name__') else str(x) "
                "for x in (LoraConfig,PeftModel,TaskType,get_peft_model)]}))"
            ),
            str(vendor.resolve()),
        ],
        text=True,
        env={**os.environ, "PYTHONPATH": str(vendor.resolve())},
    )
    probe_result = json.loads(probe)
    if (
        probe_result.get("version") != "0.20.0"
        or not str(probe_result.get("file", "")).startswith(str(vendor.resolve()))
        or len(probe_result.get("symbols", [])) != 4
    ):
        raise ValueError("PEFT vendor import probe did not bind to the frozen ZIP")
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "PEFT_VENDOR_REMOTE_BRIDGE",
        "source_bundle_sha256": expected_source_sha256,
        "source_member": EXPECTED_MEMBER,
        "vendor_zip_sha256": sha256_bytes(payload),
        "vendor_zip_size": len(payload),
        "vendor_zip_members": len(normalized),
        "metadata_version": "0.20.0",
        "import_probe": probe_result,
        "labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT_PEFT_VENDOR_BRIDGE",
    }
    result["bridge_sha256"] = canonical_sha256(result)
    (output / "acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    if len(args.expected_source_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.expected_source_sha256
    ):
        raise ValueError("source SHA must be 64 lowercase hexadecimal characters")
    print(
        json.dumps(
            bridge(
                args.source,
                args.output,
                expected_source_sha256=args.expected_source_sha256,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
