from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


BOOTSTRAP = Path("/work/code/minicpm_transformers_bootstrap.py")
TRAIN_SCRIPT = Path("/work/code/qwen3vl_lora_holdout.py")
OUTPUT_ROOT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
SCREEN_FOLDS = (0, 3)


def run_fold(fold: int) -> None:
    environment = os.environ.copy()
    environment["FULL_TRAIN"] = "0"
    environment["HOLDOUT_FOLD"] = str(fold)
    environment["ECUP_OUTPUT_DIR"] = str(OUTPUT_ROOT / f"fold_{fold}")
    print(f"starting fold={fold}", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(BOOTSTRAP), str(TRAIN_SCRIPT)],
        env=environment,
        check=True,
    )
    print(f"completed fold={fold}", flush=True)


def main() -> None:
    for fold in SCREEN_FOLDS:
        run_fold(fold)
    temporary_archive = Path("/work/minicpm_screen_output")
    archive = Path(shutil.make_archive(str(temporary_archive), "zip", OUTPUT_ROOT))
    shutil.move(str(archive), OUTPUT_ROOT / "output_bundle.zip")
    print("completed predeclared folds 0 and 3", flush=True)


if __name__ == "__main__":
    main()
