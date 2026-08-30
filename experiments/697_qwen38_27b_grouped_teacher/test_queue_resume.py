from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


QUEUE = Path(__file__).with_name("queue_remaining.sh")


class QueueResumeTest(unittest.TestCase):
    def test_lane_a_skips_completed_fold_and_reaches_failed_successor(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            local = base / ".local"
            for fold in (0, 1):
                output = local / f"output-f{fold}"
                output.mkdir(parents=True)
                (output / "output_contract.json").write_text("{}\n")

            stub = base / "python-stub"
            calls = base / "calls.txt"
            stub.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
shift
fold=
output=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fold) fold=$2; shift 2 ;;
    --output-dir) output=$2; shift 2 ;;
    *) shift ;;
  esac
done
echo "$fold" >> "$EXP697_STUB_CALLS"
mkdir -p "$output"
printf '{}\\n' > "$output/output_contract.json"
""",
                encoding="utf-8",
            )
            stub.chmod(0o755)
            environment = {
                **os.environ,
                "EXP697_BASE": str(base),
                "EXP697_PYTHON": str(stub),
                "EXP697_STUB_CALLS": str(calls),
            }
            result = subprocess.run(
                ["bash", str(QUEUE), "lane-a"],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertIn("fold 1 already has a completed contract", result.stdout)
            self.assertEqual(calls.read_text(encoding="utf-8").splitlines(), ["4"])
            self.assertTrue((local / "output-f4" / "output_contract.json").is_file())


if __name__ == "__main__":
    unittest.main()
