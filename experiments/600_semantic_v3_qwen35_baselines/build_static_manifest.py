from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import build_static_protocol_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the static experiment-600 protocol manifest."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_static_protocol_manifest(args.data.resolve(), args.folds.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
