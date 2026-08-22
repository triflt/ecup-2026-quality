from __future__ import annotations

import os
import subprocess
import sys


target = "/work/transformers455"
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--target", target,
    "transformers==4.55.0", "einops==0.8.1", "sentencepiece", "protobuf",
])
environment = os.environ.copy()
environment["PYTHONPATH"] = target
os.execvpe(sys.executable, [sys.executable, "-u", "/work/code/paddleocr_smoke_455.py"], environment)
