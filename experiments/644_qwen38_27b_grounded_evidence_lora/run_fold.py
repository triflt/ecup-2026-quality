from __future__ import annotations

import json
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from train_lora import parser_for, run

if __name__ == "__main__":
    print(
        json.dumps(
            run("644", parser_for("644").parse_args()), ensure_ascii=False, indent=2, sort_keys=True
        )
    )
