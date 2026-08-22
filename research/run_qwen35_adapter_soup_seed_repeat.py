from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

TRAINER = Path("/work/code/qwen35_bad_family_diverse_positives_lora.py")
SCORER = Path("/work/code/qwen35_adapter_soup_seed_repeat_oof.py")
OUTPUT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
ORIGINAL = Path("/work/input/original_seed_adapter.zip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_original() -> None:
    if ORIGINAL.exists():
        raise FileExistsError(f"refusing to reuse original adapter: {ORIGINAL}")
    temporary = ORIGINAL.with_suffix(".download")
    urllib.request.urlretrieve(os.environ["ORIGINAL_ADAPTER_URL"], temporary)
    expected = os.environ["ORIGINAL_ADAPTER_SHA256"]
    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink()
        raise ValueError(f"original adapter checksum mismatch: {actual} != {expected}")
    temporary.rename(ORIGINAL)


def main() -> None:
    fold = int(os.environ["HOLDOUT_FOLD"])
    if fold not in {0, 3}:
        raise ValueError(f"fold outside frozen screen: {fold}")
    if int(os.environ.get("SEED", "0")) != 31415:
        raise ValueError("seed must remain frozen at 31415")
    download_original()

    train_output = OUTPUT / f"fold_{fold}" / "specialist"
    environment = os.environ.copy()
    environment.update({
        "FULL_TRAIN": "0",
        "HOLDOUT_FOLD": str(fold),
        "FAMILY_BALANCE_FLAMMABLE": "0",
        "FAMILY_DIVERSE_BAD_POSITIVES": "1",
        "FAMILY_DIVERSE_FLAMMABLE_NEGATIVES": "0",
        "ECUP_OUTPUT_DIR": str(train_output),
    })
    subprocess.run([sys.executable, "-u", str(TRAINER)], env=environment, check=True)

    environment["ECUP_OUTPUT_DIR"] = str(OUTPUT / f"fold_{fold}")
    environment["SPECIALIST_ADAPTER"] = str(train_output / "adapter.zip")
    subprocess.run([sys.executable, "-u", str(SCORER)], env=environment, check=True)

    archive = Path(shutil.make_archive(
        f"/work/qwen_soup_repeat_fold_{fold}", "zip", OUTPUT / f"fold_{fold}"
    ))
    destination = OUTPUT / f"output_bundle_fold_{fold}.zip"
    shutil.move(str(archive), destination)
    print(f"saved output bundle: {destination}", flush=True)


if __name__ == "__main__":
    main()
