from __future__ import annotations

"""Project the locked 600-row inference probe to both evaluation sizes."""

import argparse
import json
from pathlib import Path

from resolution_contract import (
    PRIVATE_LIMIT_MINUTES,
    PRIVATE_ROWS,
    PUBLIC_LIMIT_MINUTES,
    PUBLIC_ROWS,
    RUNTIME_PROBE_ROWS,
)


def project_runtime(elapsed_seconds: float, *, observed_rows: int = RUNTIME_PROBE_ROWS) -> dict:
    if observed_rows != RUNTIME_PROBE_ROWS:
        raise ValueError(f"runtime gate requires exactly {RUNTIME_PROBE_ROWS} rows")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed seconds must be positive")
    seconds_per_row = elapsed_seconds / observed_rows
    public_minutes = seconds_per_row * PUBLIC_ROWS / 60.0
    private_minutes = seconds_per_row * PRIVATE_ROWS / 60.0
    gates = {
        "public_within_16_minutes": public_minutes <= PUBLIC_LIMIT_MINUTES,
        "private_within_32_minutes": private_minutes <= PRIVATE_LIMIT_MINUTES,
    }
    return {
        "status": "runtime_gate_passed" if all(gates.values()) else "runtime_gate_failed",
        "observed_rows": observed_rows,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_row": seconds_per_row,
        "projected_public_rows": PUBLIC_ROWS,
        "projected_public_minutes": public_minutes,
        "projected_private_rows": PRIVATE_ROWS,
        "projected_private_minutes": private_minutes,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--observed-rows", type=int, default=RUNTIME_PROBE_ROWS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = project_runtime(args.elapsed_seconds, observed_rows=args.observed_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

