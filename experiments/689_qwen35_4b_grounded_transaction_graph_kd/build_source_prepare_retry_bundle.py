"""Build the exp689 source-PREPARE bundle with the accepted transport fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_source_prepare_bundle as base

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
FILES = (
    f"{EXPERIMENT}/extract_source_archive_transport.py",
    f"{EXPERIMENT}/prepare_source_universe.py",
    f"{EXPERIMENT}/source_prepare_spec_v1.json",
    f"{EXPERIMENT}/verify_source_prepare.py",
)


def build(repo: Path, revision: str, output_dir: Path) -> dict[str, object]:
    return base.build(
        repo,
        revision,
        output_dir,
        files=FILES,
        manifest_schema="exp689_source_prepare_bundle_manifest_v2_transport",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
