from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from ecup_quality.experiments.runner import run_entrypoint

    raise SystemExit(run_entrypoint(Path(__file__).with_name("experiment.toml"), sys.argv[1:]))
