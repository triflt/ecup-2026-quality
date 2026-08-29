from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


RUN = Path("/tmp/exp711-screen-fold0-retry1")
OUTPUT = Path("/tmp/exp711-evals-fold0-retry1")
ROOT = Path("/remote_compute/home/repos/quality")
PYTHON = Path("/remote_compute/home/.venv-exp699/bin/python")
EVALUATOR = ROOT / "experiments/711_qwen_gemma_peft_ablation/evaluate_epoch.py"
DATA = Path("/remote_compute/home/data/exp699/solution140_full_support_sources/data.csv")
EVAL = Path("/remote_compute/home/data/exp699/eval")


def evaluate(predictions: Path, destination: Path, replace_leg: str) -> None:
    subprocess.run(
        [
            str(PYTHON), str(EVALUATOR),
            "--predictions", str(predictions),
            "--replace-leg", replace_leg,
            "--fold", "0",
            "--data", str(DATA),
            "--four-head", str(EVAL / "four_head_oof.npz"),
            "--qwen3vl", str(EVAL / "qwen3vl_baseline.npz"),
            "--qwen35", str(EVAL / "qwen35_baseline.npz"),
            "--output", str(destination),
        ],
        check=True,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    expected = 8 * 5
    while True:
        produced = 0
        for arm in sorted(path for path in RUN.iterdir() if path.is_dir()):
            replace_leg = "qwen35" if arm.name.startswith("qwen35_4b_") else "qwen3vl"
            for epoch in range(1, 6):
                predictions = arm / "output" / f"epoch_{epoch}" / "lora_holdout_predictions.csv"
                destination = OUTPUT / f"{arm.name}_epoch_{epoch}.json"
                if predictions.is_file() and not destination.exists():
                    evaluate(predictions, destination, replace_leg)
                if destination.is_file():
                    produced += 1
        print(json.dumps({"evaluated": produced, "expected": expected}), flush=True)
        if produced == expected:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
