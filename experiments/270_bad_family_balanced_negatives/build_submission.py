from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    config = Path(__file__).with_name("experiment.toml")
    command = [
        sys.executable,
        str(ROOT / "tools" / "build_experiment_submission.py"),
        "--config",
        str(config),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)
