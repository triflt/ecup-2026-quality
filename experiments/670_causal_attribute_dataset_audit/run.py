from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("export-baseline", "build", "score"))
    args, remainder = parser.parse_known_args()
    script = {
        "export-baseline": "export_baseline.py",
        "build": "build_audit.py",
        "score": "evaluate_audit.py",
    }[args.command]
    return subprocess.run([sys.executable, str(HERE / script), *remainder], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
