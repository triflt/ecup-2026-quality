from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

_ORIGINAL_SYS_PATH = list(sys.path)
from run_fold import RESUME_SCHEMA, atomic_torch_save, load_resume_checkpoint
sys.path[:] = _ORIGINAL_SYS_PATH


class ResumeCheckpointTest(unittest.TestCase):
    def payload(self, contract: dict, *, next_offset: int = 4) -> dict:
        return {
            "schema_version": RESUME_SCHEMA,
            "contract": contract,
            "next_offset": next_offset,
            "micro_step": next_offset // 2,
            "optimizer_steps": 1,
            "adapter_state": {"lora": torch.tensor([1.0])},
            "optimizer_state": {"state": {}, "param_groups": []},
            "scheduler_state": {"last_epoch": 1},
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [],
            "elapsed_seconds": 12.5,
        }

    def test_atomic_round_trip_and_contract_binding(self) -> None:
        contract = {"fold": 1, "train_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "fold1.resume.pt"
            atomic_torch_save(self.payload(contract), path)
            loaded = load_resume_checkpoint(
                path, contract, micro_batch=2, rows=8
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["next_offset"], 4)
            self.assertFalse(any(Path(root).glob("*.tmp")))
            with self.assertRaisesRegex(ValueError, "training contract mismatch"):
                load_resume_checkpoint(
                    path, {"fold": 2}, micro_batch=2, rows=8
                )

    def test_rejects_misaligned_progress(self) -> None:
        contract = {"fold": 1}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "fold1.resume.pt"
            atomic_torch_save(self.payload(contract, next_offset=3), path)
            with self.assertRaisesRegex(ValueError, "offset is invalid"):
                load_resume_checkpoint(
                    path, contract, micro_batch=2, rows=8
                )


if __name__ == "__main__":
    unittest.main()
