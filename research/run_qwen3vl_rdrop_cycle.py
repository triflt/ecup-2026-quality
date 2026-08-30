from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

TRAIN_SCRIPT = Path("/work/code/qwen3vl_rdrop_lora.py")
OUTPUT_ROOT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
AUTHORIZED_SCREEN_FOLDS = {0, 3}


def run_stage(stage: str) -> None:
    environment = os.environ.copy()
    environment["TRAINING_MODE"] = "hard"
    environment["FULL_TRAIN"] = "0"
    environment["RDROP_ALPHA"] = "1.0"
    environment["ECUP_OUTPUT_DIR"] = str(OUTPUT_ROOT / stage)
    if not stage.startswith("fold_"):
        raise ValueError(f"screen cycle accepts fold stages only: {stage}")
    fold = int(stage.removeprefix("fold_"))
    if fold not in AUTHORIZED_SCREEN_FOLDS:
        raise ValueError(f"fold {fold} is not authorized by the reject-only screen")
    environment["HOLDOUT_FOLD"] = str(fold)
    print(f"starting stage={stage}", flush=True)
    subprocess.run([sys.executable, "-u", str(TRAIN_SCRIPT)], env=environment, check=True)
    print(f"completed stage={stage}", flush=True)


def main() -> None:
    stages = [
        value.strip()
        for value in os.environ.get("STAGES", "fold_0,fold_3").split(",")
        if value.strip()
    ]
    if not stages:
        raise ValueError("STAGES is empty")
    for stage in stages:
        run_stage(stage)
    suffix = os.environ.get("BUNDLE_SUFFIX", "screen")
    temporary = Path(f"/work/qwen_rdrop_{suffix}")
    archive = Path(shutil.make_archive(str(temporary), "zip", OUTPUT_ROOT))
    destination = OUTPUT_ROOT / f"output_bundle_{suffix}.zip"
    shutil.move(str(archive), destination)
    print(f"saved output bundle: {destination}", flush=True)


if __name__ == "__main__":
    main()
