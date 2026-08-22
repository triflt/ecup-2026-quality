from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    raise SystemExit(
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "preprocess_experiment.py"), *sys.argv[1:]],
            cwd=ROOT,
            check=False,
        ).returncode
    )
