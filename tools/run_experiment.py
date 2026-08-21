from __future__ import annotations

import argparse

from ecup_quality.experiments.runner import run_entrypoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, remainder = parser.parse_known_args()
    raise SystemExit(run_entrypoint(args.config, remainder))


if __name__ == "__main__":
    main()
