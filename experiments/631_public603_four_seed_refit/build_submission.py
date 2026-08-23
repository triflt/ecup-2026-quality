from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_production_manifest() -> None:
    manifest_path = HERE / "results" / "production_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("production manifest is absent; training/runtime gates are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != 631 or manifest.get("parent_experiment") != 603:
        raise SystemExit("production manifest identity mismatch")
    if manifest.get("runtime", {}).get("accepted") is not True:
        raise SystemExit("runtime gate is not accepted")
    adapters = manifest.get("adapters", {})
    expected_names = {
        "42": "adapter_qwen35",
        "31415": "adapter_qwen35_seed31415",
        "271828": "adapter_qwen35_seed271828",
        "161803": "adapter_qwen35_seed161803",
    }
    if set(adapters) != set(expected_names):
        raise SystemExit("production manifest must bind exactly four frozen seeds")
    for seed, directory in expected_names.items():
        path = HERE / "submission" / directory / "adapter_model.safetensors"
        if not path.is_file() or sha256_file(path) != adapters[seed].get("sha256"):
            raise SystemExit(f"adapter integrity mismatch for seed {seed}")


if __name__ == "__main__":
    verify_production_manifest()
    config = Path(__file__).with_name("experiment.toml")
    command = [
        sys.executable,
        str(ROOT / "tools" / "build_experiment_submission.py"),
        "--config",
        str(config),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)
