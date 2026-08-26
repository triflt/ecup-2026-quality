"""Build the minimal exp689 fold0 validation-binding diagnostic bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_source_prepare_bundle as base
import diagnose_fold0_validation_binding as diagnostic

FILES = tuple(sorted(diagnostic.BUNDLE_PATHS))


def build(repo: Path, revision: str, output_dir: Path) -> dict[str, object]:
    return base.build(
        repo,
        revision,
        output_dir,
        files=FILES,
        manifest_schema=diagnostic.BUNDLE_SCHEMA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
