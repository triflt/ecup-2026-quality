from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "validation/connected_family_repeated_v1/rows.csv"
MANIFEST = ROOT / "validation/connected_family_repeated_v1/manifest.json"


def test_repeats_keep_each_safe_component_in_one_fold() -> None:
    frame = pd.read_csv(ROWS, dtype={"id": str, "connected_component": str})
    safe = frame.safe_for_selection.astype(bool)
    repeat_columns = [column for column in frame if column.startswith("repeat_")]
    assert repeat_columns
    for column in repeat_columns:
        assert (frame.loc[safe, column] >= 0).all()
        assert (frame.loc[~safe, column] == -1).all()
        counts = frame.loc[safe].groupby(
            ["category", "connected_component"]
        )[column].nunique()
        assert counts.max() == 1


def test_repeat_manifest_is_predeclared() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["seeds"] == [17, 31415, 20260822]
    assert manifest["n_splits"] == 5
    assert manifest["safe_rows"] == 11207
    for repeat in manifest["repeats"].values():
        for category in repeat["categories"].values():
            assert category["component_max_fold_count"] == 1
