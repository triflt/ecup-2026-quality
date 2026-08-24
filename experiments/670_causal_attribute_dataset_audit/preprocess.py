from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(
        subprocess.run(
            [sys.executable, str(HERE / "build_audit.py"), *sys.argv[1:]], check=False
        ).returncode
    )
