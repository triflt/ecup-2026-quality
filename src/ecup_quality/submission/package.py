from __future__ import annotations

import zipfile
from pathlib import Path

EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache", ".DS_Store"}


def build_submission_archive(source: Path, output: Path) -> Path:
    """Build a local submission ZIP. The archive is ignored by Git by policy."""
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not (source / "run.py").exists() or not (source / "metadata.json").exists():
        raise ValueError("submission source must contain run.py and metadata.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            archive.write(path, path.relative_to(source).as_posix())
    return output
