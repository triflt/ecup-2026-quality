from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/540_qwen35_mixed_family_bad_pairwise/parent_selector.py"
SPEC = importlib.util.spec_from_file_location("_exp560_exact_parent_selector", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError("exact dependency-light parent selector is unavailable")
_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_module)
select_parent_training_records = _module.select_parent_training_records

__all__ = ["select_parent_training_records"]
