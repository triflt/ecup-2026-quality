from __future__ import annotations

import tomllib
from pathlib import Path


def main() -> None:
    for path in sorted(Path("experiments").glob("*/experiment.toml")):
        raw = tomllib.loads(path.read_text(encoding="utf-8"))["experiment"]
        print(f"{raw['id']:<4} {raw['status']:<12} {raw['title']} ({path.parent})")


if __name__ == "__main__":
    main()
