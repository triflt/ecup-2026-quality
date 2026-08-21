from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


submission = Path("/work/submission")
shutil.unpack_archive(
    "/work/lora-artifacts/adapter.zip", submission / "adapter", "zip"
)
shutil.unpack_archive(
    "/work/peft-artifacts/peft-0.20.0.zip", submission / "vendor", "zip"
)
subprocess.run(
    [
        sys.executable,
        "-u",
        str(submission / "run.py"),
        "--test_data_path",
        "/work/test.csv",
        "--output_path",
        "/work/output/submission.csv",
    ],
    cwd=submission,
    check=True,
)
