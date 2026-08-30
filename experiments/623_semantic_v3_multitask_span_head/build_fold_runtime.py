from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import build_fold_runtime


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a label-isolated experiment-623 fold runtime."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--evidence-manifest", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--image-manifest", required=True, type=Path)
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(5))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build_fold_runtime(
        data_path=args.data.resolve(),
        folds_path=args.folds.resolve(),
        evidence_manifest_path=args.evidence_manifest.resolve(),
        selector_path=args.selector.resolve(),
        image_manifest_path=args.image_manifest.resolve(),
        outer_fold=args.outer_fold,
        output_dir=args.output_dir.resolve(),
        enforce_frozen_hashes=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
