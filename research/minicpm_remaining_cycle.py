from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


BOOTSTRAP = Path("/work/code/minicpm_transformers_bootstrap.py")
TRAIN_SCRIPT = Path("/work/code/qwen3vl_lora_holdout.py")
OUTPUT_ROOT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
REMAINING_FOLDS = (1, 2, 4)


def run_stage(name: str, fold: int | None) -> None:
    environment = os.environ.copy()
    environment["ECUP_OUTPUT_DIR"] = str(OUTPUT_ROOT / name)
    if fold is None:
        environment["FULL_TRAIN"] = "1"
        environment.pop("HOLDOUT_FOLD", None)
    else:
        environment["FULL_TRAIN"] = "0"
        environment["HOLDOUT_FOLD"] = str(fold)
    print(f"starting stage={name}", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(BOOTSTRAP), str(TRAIN_SCRIPT)],
        env=environment,
        check=True,
    )
    print(f"completed stage={name}", flush=True)


def main() -> None:
    for fold in REMAINING_FOLDS:
        run_stage(f"fold_{fold}", fold)
    run_stage("full", None)
    archive = Path(shutil.make_archive("/work/minicpm_remaining_output", "zip", OUTPUT_ROOT))
    shutil.move(str(archive), OUTPUT_ROOT / "output_bundle.zip")
    print("completed folds 1, 2, 4 and full refit", flush=True)


if __name__ == "__main__":
    main()
