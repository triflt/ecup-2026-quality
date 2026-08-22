from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    paths = {
        "data": args.data,
        "oof": args.oof,
        "image_manifest": args.image_manifest,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    expected = {
        name: manifest["sources"][name]["sha256"] for name in sorted(paths)
    }
    if observed != expected:
        raise ValueError(f"frozen source mismatch: observed={observed} expected={expected}")
    print(json.dumps({"status": "ok", "sha256": observed}, indent=2))


if __name__ == "__main__":
    main()
