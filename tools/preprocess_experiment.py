from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    required = {"id", "category", "label", "name", "description"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print({"rows": len(frame), "columns": list(frame.columns), "output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
