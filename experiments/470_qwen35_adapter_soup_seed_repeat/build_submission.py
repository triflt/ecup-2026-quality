from __future__ import annotations

import json
from pathlib import Path

RESULT = Path(__file__).with_name("results") / "metrics.json"


def main() -> None:
    metrics = json.loads(RESULT.read_text())
    if metrics.get("status") != "accepted_full_cycle":
        raise SystemExit(
            "Experiment 470 is a reject-only two-fold screen. A submission can be "
            "built only by a separately versioned full-cycle experiment after acceptance."
        )
    raise SystemExit("No full-cycle submission builder has been registered.")


if __name__ == "__main__":
    main()
