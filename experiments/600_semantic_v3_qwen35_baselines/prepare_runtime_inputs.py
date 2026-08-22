from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import prepare_runtime_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare scoped experiment-600 runtime inputs.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--image-manifest", required=True, type=Path)
    parser.add_argument("--robust-base", required=True, type=Path)
    parser.add_argument("--robust-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = prepare_runtime_inputs(
        data_path=args.data.resolve(),
        folds_path=args.folds.resolve(),
        image_manifest_path=args.image_manifest.resolve(),
        robust_base_path=args.robust_base.resolve(),
        robust_report_path=args.robust_report.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
