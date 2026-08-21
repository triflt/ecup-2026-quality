from __future__ import annotations

import re
import subprocess
from pathlib import Path

FORBIDDEN_CONTENT = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"(?:private-bank|private-storage|private-tech)",
        r"private-object-store\.",
        r"artifactory\.",
        r"model-registry(?:-old)?\.",
        r"X-Amz-(?:Credential|Signature|Security-Token)",
        r"\b(?:tenant|project)[_-]?(?:id|name)?\s*[:=]\s*['\"][^$<{]",
    ]
]
FORBIDDEN_SUFFIXES = {".safetensors", ".joblib", ".pkl", ".zip", ".npz", ".npy", ".pt", ".pth"}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True
    )
    return [Path(line) for line in output.splitlines() if line]


def main() -> None:
    violations = []
    for path in tracked_files():
        if path == Path("tools/check_publish_safety.py"):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"binary artifact is tracked: {path}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in FORBIDDEN_CONTENT:
            if pattern.search(text):
                violations.append(f"forbidden internal reference in {path}: {pattern.pattern}")
                break
    if violations:
        print("\n".join(violations))
        raise SystemExit(2)
    print(f"publish safety: PASS ({len(tracked_files())} tracked files checked)")


if __name__ == "__main__":
    main()
