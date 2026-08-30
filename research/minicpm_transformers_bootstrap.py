from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path


VENDOR = Path(os.environ.get("TRANSFORMERS_VENDOR", "/work/vendor/transformers57"))
VERSION = os.environ.get("TRANSFORMERS_VERSION", "5.7.0")
INSTALL_DEPS = os.environ.get("TRANSFORMERS_INSTALL_DEPS", "0") == "1"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: minicpm_transformers_bootstrap.py SCRIPT [ARGS...]")
    if not (VENDOR / "transformers").is_dir():
        VENDOR.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(VENDOR),
            "--no-cache-dir",
        ]
        if not INSTALL_DEPS:
            command.append("--no-deps")
        command.append(f"transformers=={VERSION}")
        subprocess.run(command, check=True)
    sys.path.insert(0, str(VENDOR))
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
