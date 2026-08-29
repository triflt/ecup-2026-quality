from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_gradient_control import TRAINING_MODES, verify_training_artifact
from train_gradient_control import parser as training_parser


def parser() -> argparse.ArgumentParser:
    result = training_parser()
    result.description = "Independently verify a frozen experiment-688 training artifact."
    result.add_argument("--output", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.mode not in TRAINING_MODES:
        raise ValueError("unknown exp688 mode")
    value = verify_training_artifact(args)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        expected = args.output_dir / "acceptance.json"
        if args.output.resolve() != expected.resolve():
            raise ValueError("exp688 acceptance must be output-dir/acceptance.json")
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") != payload:
                raise FileExistsError("refusing to overwrite a different acceptance")
        else:
            args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
