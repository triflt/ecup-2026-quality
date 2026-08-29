from __future__ import annotations

import tomllib
from pathlib import Path


def main() -> None:
    for path in sorted(Path("experiments").glob("*/experiment.toml")):
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        raw = document.get("experiment", document)
        experiment_id = str(raw.get("id", path.parent.name.split("_", 1)[0]))
        status = str(raw.get("status", "unspecified"))
        title = str(raw.get("title") or raw.get("name") or raw.get("slug") or path.parent.name)
        print(f"{experiment_id:<4} {status:<36} {title} ({path.parent})")


if __name__ == "__main__":
    main()
