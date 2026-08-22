from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


TRAIN_SCRIPT = Path("/work/code/qwen35_family_balanced_lora.py")
OUTPUT_ROOT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))


def run_stage(name: str, *, fold: int | None) -> None:
    environment = os.environ.copy()
    environment["FAMILY_BALANCE_FLAMMABLE"] = "1"
    environment["ECUP_OUTPUT_DIR"] = str(OUTPUT_ROOT / name)
    if fold is None:
        environment["FULL_TRAIN"] = "1"
        environment.pop("HOLDOUT_FOLD", None)
    else:
        environment["FULL_TRAIN"] = "0"
        environment["HOLDOUT_FOLD"] = str(fold)
    print(f"starting stage={name}", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(TRAIN_SCRIPT)],
        env=environment,
        check=True,
    )
    print(f"completed stage={name}", flush=True)


def main() -> None:
    for fold in range(5):
        run_stage(f"fold_{fold}", fold=fold)
    run_stage("full", fold=None)


if __name__ == "__main__":
    main()
