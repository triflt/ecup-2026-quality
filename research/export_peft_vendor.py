from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


target = Path("/work/vendor")
output = Path("/work/output")
target.mkdir(parents=True, exist_ok=True)
output.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
        "--no-cache-dir",
        "--no-deps",
        "peft==0.20.0",
    ],
    check=True,
)
shutil.make_archive(str(output / "peft-0.20.0"), "zip", target)
print(output / "peft-0.20.0.zip", flush=True)
