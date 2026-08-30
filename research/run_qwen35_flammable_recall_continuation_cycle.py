from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import hashlib
import urllib.request
import zipfile
from pathlib import Path


TRAIN_SCRIPT = Path("/work/code/qwen35_flammable_recall_continuation.py")
OUTPUT_ROOT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
PARENT_ROOT = Path(os.environ.get("PARENT_ADAPTER_ROOT", "/work/input/parent_adapters"))


def extract_parent(stage: str) -> Path:
    archive = PARENT_ROOT / f"{stage}.zip"
    if not archive.is_file():
        url = os.environ.get("PARENT_ADAPTER_URL")
        expected_sha256 = os.environ.get("PARENT_ADAPTER_SHA256")
        if not url or not expected_sha256:
            raise FileNotFoundError(
                f"parent adapter is absent and URL/checksum are unset: {archive}"
            )
        PARENT_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".download")
        if temporary.exists():
            raise FileExistsError(f"refusing to reuse partial parent download: {temporary}")
        urllib.request.urlretrieve(url, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != expected_sha256:
            temporary.unlink()
            raise ValueError(
                f"parent adapter checksum mismatch: {digest} != {expected_sha256}"
            )
        temporary.rename(archive)
    destination = Path("/work") / f"parent_{stage}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite parent extraction: {destination}")
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise ValueError(f"parent adapter CRC failure: {bad}")
        for info in bundle.infolist():
            path = Path(info.filename)
            mode = info.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                raise ValueError(f"unsafe parent adapter member: {info.filename}")
        bundle.extractall(destination)
    if not (destination / "adapter_config.json").is_file():
        raise ValueError(f"parent adapter lacks adapter_config.json: {stage}")
    return destination


def run_stage(stage: str) -> None:
    environment = os.environ.copy()
    environment["STAGE"] = stage
    environment["FAMILY_BALANCE_FLAMMABLE"] = "0"
    environment["FAMILY_DIVERSE_BAD_POSITIVES"] = "1"
    environment["FAMILY_DIVERSE_FLAMMABLE_NEGATIVES"] = "0"
    environment["PARENT_ADAPTER"] = str(extract_parent(stage))
    environment["ECUP_OUTPUT_DIR"] = str(OUTPUT_ROOT / stage)
    if stage == "full":
        environment["FULL_TRAIN"] = "1"
        environment.pop("HOLDOUT_FOLD", None)
    elif stage.startswith("fold_"):
        environment["FULL_TRAIN"] = "0"
        environment["HOLDOUT_FOLD"] = stage.removeprefix("fold_")
    else:
        raise ValueError(f"unknown stage: {stage}")
    print(f"starting stage={stage}", flush=True)
    subprocess.run([sys.executable, "-u", str(TRAIN_SCRIPT)], env=environment, check=True)
    print(f"completed stage={stage}", flush=True)


def main() -> None:
    stages = [
        value.strip()
        for value in os.environ.get("STAGES", "fold_0,fold_3").split(",")
        if value.strip()
    ]
    for stage in stages:
        run_stage(stage)
    suffix = os.environ.get("BUNDLE_SUFFIX", "screen")
    temporary_archive = Path(f"/work/qwen_recall_{suffix}")
    archive_path = Path(shutil.make_archive(str(temporary_archive), "zip", OUTPUT_ROOT))
    destination = OUTPUT_ROOT / f"output_bundle_{suffix}.zip"
    shutil.move(str(archive_path), destination)
    print(f"saved output bundle: {destination}", flush=True)


if __name__ == "__main__":
    main()
