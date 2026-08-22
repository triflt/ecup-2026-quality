#!/usr/bin/env bash
set -euo pipefail
uv run --no-sync python experiments/320_bad_regulatory_evidence/run.py "$@"
