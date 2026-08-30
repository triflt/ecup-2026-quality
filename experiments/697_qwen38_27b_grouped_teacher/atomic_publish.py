from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_output_directory(output_dir: Path):
    """Publish a complete directory or leave no final artifact."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite target output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    try:
        yield staging
        staging.replace(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
