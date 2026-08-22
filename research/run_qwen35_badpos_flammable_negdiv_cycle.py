from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


TRAIN_SCRIPT = Path("/work/code/qwen35_bad_family_diverse_positives_lora.py")
OUTPUT_ROOT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))


def run_stage(name: str) -> None:
    environment = os.environ.copy()
    environment["FAMILY_BALANCE_FLAMMABLE"] = "0"
    environment["FAMILY_DIVERSE_BAD_POSITIVES"] = "1"
    environment["FAMILY_DIVERSE_FLAMMABLE_NEGATIVES"] = "1"
    environment["ECUP_OUTPUT_DIR"] = str(OUTPUT_ROOT / name)
    if name == "full":
        environment["FULL_TRAIN"] = "1"
        environment.pop("HOLDOUT_FOLD", None)
    elif name.startswith("fold_"):
        environment["FULL_TRAIN"] = "0"
        environment["HOLDOUT_FOLD"] = name.removeprefix("fold_")
    else:
        raise ValueError(f"unknown stage: {name}")
    print(f"starting stage={name}", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(TRAIN_SCRIPT)],
        env=environment,
        check=True,
    )
    print(f"completed stage={name}", flush=True)


def main() -> None:
    stages = [
        value.strip()
        for value in os.environ.get(
            "STAGES", "fold_0,fold_1,fold_2,fold_3,fold_4,full"
        ).split(",")
        if value.strip()
    ]
    for stage in stages:
        run_stage(stage)
    suffix = os.environ.get("BUNDLE_SUFFIX", "all")
    temporary_archive = Path(f"/work/qwen_negdiv_{suffix}")
    archive_path = Path(shutil.make_archive(
        str(temporary_archive), "zip", OUTPUT_ROOT
    ))
    destination = OUTPUT_ROOT / f"output_bundle_{suffix}.zip"
    shutil.move(str(archive_path), destination)
    print(f"saved output bundle: {destination}", flush=True)


if __name__ == "__main__":
    main()
