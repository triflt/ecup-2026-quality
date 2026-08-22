from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(PACKAGE_ROOT))

from evidence_grounding.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
