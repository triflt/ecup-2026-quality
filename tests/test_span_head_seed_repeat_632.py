from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/632_span_head_seed_repeat/evaluate_screen.py"
SPEC = importlib.util.spec_from_file_location("exp632_screen", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_wrapper_changes_only_candidate_identity_and_seed() -> None:
    parent = MODULE.load_parent()
    assert parent.CANDIDATE_EXPERIMENT_ID == "632"
    assert parent.CANDIDATE_SEED == 31415
    assert parent.RUNTIME_EXPERIMENT_ID == "623"
    assert parent.SCREEN_FOLDS == (0, 3)
