from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "validation/connected_family_guard_v2/rows.csv"
MANIFEST = ROOT / "validation/connected_family_guard_v2/manifest.json"


def test_connected_guard_has_no_safe_cross_fold_component() -> None:
    frame = pd.read_csv(ROWS, dtype={"id": str, "connected_component": str})
    safe = frame[frame.safe_for_selection.astype(bool)]
    fold_counts = safe.groupby(["category", "connected_component"]).fold.nunique()
    assert fold_counts.max() == 1
    assert (fold_counts > 1).sum() == 0
    assert not frame.id.duplicated().any()


def test_connected_guard_manifest_matches_rows() -> None:
    frame = pd.read_csv(ROWS, dtype={"id": str})
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["rows"] == len(frame)
    assert manifest["safe_rows"] == int(frame.safe_for_selection.astype(bool).sum())
    assert manifest["unsafe_rows"] == int((~frame.safe_for_selection.astype(bool)).sum())
    assert manifest["invariants"]["safe_components_crossing_folds"] == 0
    assert manifest["invariants"]["all_cross_fold_components_flagged_unsafe"] is True
