from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/260_bad_family_diverse_positives/artifacts/seed_42_diverse_positives"
CANDIDATE = EXPERIMENT / "artifacts/seed_42_negdiv"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
LOCKED = EXPERIMENT / "results/locked_replacement_report.json"


def fold_paths(root: Path) -> list[Path]:
    paths = [root / f"fold_{fold}/lora_holdout_predictions.csv" for fold in range(5)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing fold predictions:\n" + "\n".join(missing)
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    parent = fold_paths(PARENT)
    candidate = fold_paths(CANDIDATE)
    command = [
        sys.executable,
        str(ROOT / "research/qwen35_locked_190_audit.py"),
        "--candidate",
        "exp260",
        *map(str, parent),
        "--candidate",
        "exp410",
        *map(str, candidate),
        "--folds",
        str(FOLDS),
        "--bootstrap",
        str(args.bootstrap),
        "--output",
        str(LOCKED),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT / "evaluate.py"),
            "--locked",
            str(LOCKED.with_suffix(".npz")),
            "--bootstrap",
            str(args.bootstrap),
        ],
        check=True,
    )
    audit = json.loads((EXPERIMENT / "results/acceptance_audit.json").read_text())
    print(json.dumps({
        "stage1_passed": audit["stage1_passed"],
        "delta_macro_f1": audit["delta_macro_f1"],
        "folds_won": audit["folds_won"],
        "flammable_false_negative_delta": audit["flammable_false_negatives"]["delta"],
        "safety_false_negative_delta": audit["safety_cohorts"]["union"]["false_negative_delta"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
