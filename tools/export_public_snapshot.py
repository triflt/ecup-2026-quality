from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def publishable_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    return [Path(value.decode("utf-8")) for value in output.split(b"\0") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"destination must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    files = publishable_files()
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, target)
    print({"destination": str(destination), "files": len(files)})


if __name__ == "__main__":
    main()
