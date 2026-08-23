from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from runtime_builder import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fold-isolated experiment-644 runtime.")
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--ocr-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--expected-ocr-sha256", action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_runtime(
        experiment_id="644",
        outer_fold=args.fold,
        data_path=args.data,
        folds_path=args.folds,
        selector_path=args.selector,
        image_manifest_path=args.image_manifest,
        evidence_manifest_path=args.evidence_manifest,
        ocr_paths=args.ocr_jsonl,
        output_dir=args.output_dir,
        expected_ocr_sha256=args.expected_ocr_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
