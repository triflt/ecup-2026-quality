"""Build the exact exp689 transport-aware PREPARE verifier bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_source_prepare_bundle as base

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
FILES = (f"{EXPERIMENT}/verify_source_prepare_retry.py",)


def build(repo: Path, revision: str, output_dir: Path) -> dict[str, object]:
    return base.build(
        repo,
        revision,
        output_dir,
        files=FILES,
        manifest_schema=(
            "exp689_source_prepare_retry_verifier_bundle_manifest_v1"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
