from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

FOLDS = (1, 2, 4)
FILES = ("runtime_audit.json", "train.jsonl", "validation.jsonl")
MANIFEST_NAME = "CONFIRMATION_SOURCE_MANIFEST.json"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build(source_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite confirmation source bundle")
    payloads: dict[str, bytes] = {}
    contracts: dict[str, str] = {}
    for fold in FOLDS:
        fold_root = source_root / f"fold{fold}"
        paths = {name: fold_root / name for name in FILES}
        if any(path.is_symlink() or not path.is_file() for path in paths.values()):
            raise ValueError(f"fold{fold} source runtime is incomplete or non-regular")
        audit = json.loads(paths["runtime_audit.json"].read_text(encoding="utf-8"))
        body = dict(audit)
        digest = body.pop("contract_sha256", None)
        if digest != canonical_sha256(body):
            raise ValueError(f"fold{fold} runtime self-hash mismatch")
        expected = {
            "experiment_id": "641",
            "outer_fold": fold,
            "decision": "GO",
            "outer_validation_occurrences": 0,
            "validation_labels_written": 0,
            "sealed_rows_written": 0,
        }
        if any(audit.get(key) != value for key, value in expected.items()):
            raise ValueError(f"fold{fold} source runtime scope mismatch")
        for name, path in paths.items():
            relative = f"runtime/fold{fold}/{name}"
            payloads[relative] = path.read_bytes()
        observed_outputs = {
            name: sha256_bytes(payloads[f"runtime/fold{fold}/{name}"])
            for name in ("train.jsonl", "validation.jsonl")
        }
        if audit.get("output_sha256") != observed_outputs:
            raise ValueError(f"fold{fold} runtime payload SHA mismatch")
        contracts[str(fold)] = digest

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_experiment_id": "641",
        "folds": list(FOLDS),
        "runtime_contract_sha256": contracts,
        "files": {
            name: {"size": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in sorted(payloads.items())
        },
        "validation_labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "decision": "ACCEPT_CONFIRMATION_SOURCE_BUNDLE",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, payload in sorted(payloads.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(manifest_payload)
        info.mode = 0o644
        info.mtime = 0
        archive.addfile(info, io.BytesIO(manifest_payload))
    return {
        "bundle_sha256": sha256_bytes(output.read_bytes()),
        "bundle_size": output.stat().st_size,
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_contract_sha256": contracts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2, sort_keys=True))
