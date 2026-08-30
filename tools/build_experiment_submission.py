from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecup_quality.experiments.runner import load_config
from ecup_quality.submission.package import build_submission_archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.submission_source is None:
        raise SystemExit(f"experiment {config.experiment_id} has no submission source")
    output = build_submission_archive(config.submission_source, args.output)
    print(output)


if __name__ == "__main__":
    main()
