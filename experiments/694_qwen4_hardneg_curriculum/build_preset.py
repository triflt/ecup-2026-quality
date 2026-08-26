from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.job_name.startswith("qwen4-hardneg-"):
        raise ValueError("job name must start with qwen4-hardneg-")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite preset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "name": args.job_name,
                "image": args.image,
                "flavor": args.flavor,
                "gpu_count": 1,
                "command": args.command,
                "submission_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
