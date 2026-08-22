from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


TRAIN_SCRIPT = Path("/work/code/qwen3vl_lora_holdout.py")
OUTPUT_ROOT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))


def run_stage(name: str, *, fold: int | None, gpu: int) -> dict[str, object]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["ECUP_OUTPUT_DIR"] = str(OUTPUT_ROOT / name)
    environment["ECUP_IMAGES"] = str(Path("/work/images") / name)
    environment["ECUP_VENDOR"] = str(Path("/work/vendor") / name)
    if fold is None:
        environment["FULL_TRAIN"] = "1"
        environment.pop("HOLDOUT_FOLD", None)
    else:
        environment["FULL_TRAIN"] = "0"
        environment["HOLDOUT_FOLD"] = str(fold)
    print(json.dumps({"event": "starting", "stage": name, "gpu": gpu}), flush=True)
    subprocess.run(
        [sys.executable, "-u", str(TRAIN_SCRIPT)],
        env=environment,
        check=True,
    )
    print(json.dumps({"event": "completed", "stage": name, "gpu": gpu}), flush=True)
    return {"stage": name, "gpu": gpu, "fold": fold}


def run_pair(specifications: list[tuple[str, int | None, int]]) -> list[dict[str, object]]:
    with ThreadPoolExecutor(max_workers=len(specifications)) as pool:
        futures = [
            pool.submit(run_stage, name, fold=fold, gpu=gpu)
            for name, fold, gpu in specifications
        ]
        return [future.result() for future in futures]


def main() -> None:
    completed = []
    completed.extend(run_pair([("fold_0", 0, 0), ("fold_1", 1, 1)]))
    completed.extend(run_pair([("fold_2", 2, 0), ("fold_3", 3, 1)]))
    completed.extend(run_pair([("fold_4", 4, 0), ("full", None, 1)]))
    archive = Path(shutil.make_archive("/work/qwen_soft_output", "zip", OUTPUT_ROOT))
    destination = OUTPUT_ROOT / "output_bundle.zip"
    shutil.move(str(archive), destination)
    print(json.dumps({"completed": completed, "bundle": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
